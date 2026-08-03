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
import { intersectSvoPrimitive, validateSvoClusterPacking } from "../lib/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import {
  BONSAI_COURTYARD_STANDARD,
  BONSAI_POND_CANOPY,
  BONSAI_SHELF_MINIATURE,
  bonsaiCanopyGrain,
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
};

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
  // is now two records a head plus the trunk. The explicit form still spends its
  // share, which is what the budget ceiling above is guarding.
  assert.ok(planBonsai(specFor(POND_FLORETS)).leafCount > 1_000,
    "the explicit form should be spending most of its share; a large drop means the culls broke");
  const aggregate = planBonsai(specFor(BONSAI_POND_CANOPY)).leafCount;
  assert.ok(aggregate < 300, `the aggregate crown published ${aggregate}; it is meant to collapse the floret field`);
  // Two records a head, not one. The cluster carries the granulation and the
  // ellipsoid core under it carries the closure — a lattice cannot do both, and
  // a crown that publishes one record a head has gone back to the ratio at which
  // it drew as a row of smooth ellipsoids. See `CANOPY_FLORET_PERIOD_SHARE`.
  const crown = leaves(planBonsai(specFor(BONSAI_POND_CANOPY)).nodes)
    .filter((node) => node.tags?.includes("crown"));
  assert.equal(crown.filter((node) => node.kind === "cluster").length, BONSAI_POND_CANOPY.lobes);
  assert.equal(crown.filter((node) => node.kind === "ellipsoid").length, BONSAI_POND_CANOPY.lobes);
  // And nothing below the crown is a cone. A hard union is C0 but not C1, so a
  // chain of them steps its shading normal at every junction and a taper renders
  // as stacked faceted pipes; every run is one swept record now.
  assert.equal(leaves(planBonsai(specFor(BONSAI_POND_CANOPY)).nodes).filter((node) => node.kind === "cone").length, 0);
});

test("the crown's granulation is fine enough to read as curd, and distinct enough to read at all", () => {
  // The two numbers the reference is read for, pinned because both were wrong in
  // a way no assertion here caught. The grain: florets across the crown's own
  // width, of which the reference shows upwards of sixty.
  const grain = bonsaiCanopyGrain(BONSAI_POND_CANOPY);
  const across = 2 * BONSAI_POND_CANOPY.crownRadius_m[0] / grain.finePeriod_m;
  assert.ok(across >= 60, `only ${across.toFixed(0)} florets span the crown; the curd has become cobbles`);

  // The hierarchy, which is what makes it curd rather than cobbles: the octave
  // stack has to span from a head-sized mass down to that grain, and each octave
  // is exactly a halving so the whole ladder is one number and a count.
  assert.ok(grain.octaves >= 3, "a cauliflower has more than two scales in it");
  assert.ok(Math.abs(grain.latticePeriod_m - grain.finePeriod_m * 2 ** (grain.octaves - 1)) < 1e-12);
  assert.ok(grain.latticePeriod_m > 0.05 * BONSAI_POND_CANOPY.crownRadius_m[0],
    "the coarsest octave is too fine to read as a mass");

  // And the ratio, which is the defect that shipped: a lattice closes at
  // `r = P*sqrt(3)/2`, and at or above that the packing is a solid whose
  // envelope clip cuts it off flush — every mass drawing as the bare ellipsoid
  // the cluster exists to avoid. Octaves make that worse rather than better,
  // because stacking n lattices of density `phi` gives `1 - (1 - phi)^n`, so the
  // ratio has to come *down* as the octave count goes up.
  const ratio = grain.floretRadius_m / grain.latticePeriod_m;
  assert.ok(ratio < 0.5, `florets are ${ratio.toFixed(2)} of the period; three octaves of that is a solid`);
  assert.ok(ratio > 0.3, `florets are ${ratio.toFixed(2)} of the period; the crown is beads scattered on its own core`);

  // The blend cannot be allowed to fill what the ratio just bought: a smooth
  // minimum rounds away everything under its own radius, and what it must leave
  // open is the gap between two neighbouring florets.
  const gap = grain.latticePeriod_m - 2 * grain.floretRadius_m;
  assert.ok(grain.blendRadius_m < 0.5 * gap,
    `the blend is ${(grain.blendRadius_m / gap).toFixed(2)} of the gap it has to leave open`);

  // Every form's grain scales with its own crown rather than being re-authored,
  // which is the point of counting florets instead of measuring them.
  for (const [name, form] of FORMS) {
    const each = bonsaiCanopyGrain(form);
    assert.ok(Math.abs(2 * form.crownRadius_m[0] / each.finePeriod_m - across) < 1e-9,
      `${name} does not carry the pond form's grain`);
  }
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

test("the buttresses are separate blades on the bank, not a collar of lumps over the water", () => {
  // Three properties, and the object failed all three in one round: a fin as
  // thick as it is tall reads as a bread roll, fins that touch all the way round
  // read as one lumpy collar, and a fin over the coping is a root growing out of
  // open water.
  for (const seed of [0x8017a1, 1, 7, 0x51ed, 0x7fff_ffff]) {
    const plan = planBonsai(specFor(BONSAI_POND_CANOPY, { seed }));
    const fins = leaves(plan.nodes).filter((node) => node.kind === "cluster" && node.field === "seeded-lobes");
    assert.ok(fins.length >= 12, `seed ${seed}: ${fins.length} fin envelopes; the chain has collapsed back to one per root`);

    // A blade. The thin axis against the tall one, which is the section a
    // buttress is read by — a quarter is the thick end of what still reads as a
    // plate, and it is what the march's own eccentricity cap allows.
    for (const fin of fins) {
      if (fin.kind !== "cluster") continue;
      const section = Math.min(fin.lobe.x, fin.lobe.y, fin.lobe.z) / fin.lobe.y;
      assert.ok(section <= 0.26, `seed ${seed}: ${fin.id} is ${section.toFixed(2)} as thick as it is tall; that is a lump`);
    }

    // Air between them, at mid-run, where a real buttress system has its deep V
    // valleys. The merge is meant to happen only inside the base.
    const mids = fins.filter((node) => node.id.endsWith("-2")).map((node) => {
      const at = node.place?.position;
      assert.ok(at, "every fin is placed");
      return {
        angle: Math.atan2(at.z, at.x),
        radius: Math.hypot(at.x, at.z),
        half: node.kind === "cluster" ? Math.min(node.lobe.x, node.lobe.y, node.lobe.z) : 0,
      };
    }).sort((a, b) => a.angle - b.angle);
    let narrowest = Infinity;
    for (let i = 0; i < mids.length; i += 1) {
      const here = mids[i], next = mids[(i + 1) % mids.length];
      const between = ((next.angle - here.angle) + 2 * Math.PI) % (2 * Math.PI);
      const subtended = Math.asin(Math.min(1, here.half / here.radius)) + Math.asin(Math.min(1, next.half / next.radius));
      narrowest = Math.min(narrowest, (between - subtended) * 180 / Math.PI);
    }
    assert.ok(narrowest > 12, `seed ${seed}: only ${narrowest.toFixed(0)} degrees of air between adjacent fins; they will fuse into a collar`);

    // And on the bank. The whole envelope, sampled round its own bounding
    // circle, against the coping's outer foot — a fin may reach the foot and no
    // further, whatever the authored `rootReach` asked for.
    for (const fin of fins) {
      const at = fin.place?.position;
      assert.ok(at, "every fin is placed");
      const reach = fin.kind === "cluster" ? Math.max(fin.lobe.x, fin.lobe.y, fin.lobe.z) : 0;
      for (let i = 0; i < 48; i += 1) {
        const angle = 2 * Math.PI * i / 48;
        const x = STAND_M[0] + at.x + reach * Math.cos(angle);
        const z = STAND_M[1] + at.z + reach * Math.sin(angle);
        assert.ok(pondVesselPlanDistance(CURVE, x, z) > VESSEL.rimHalfWidth_m,
          `seed ${seed}: ${fin.id} reaches over the coping`);
      }
    }
  }
});

test("a buttress fin is eccentric enough to be a plate and not so eccentric the march misses it", () => {
  // The trade this field is here for. A buttress root is a fin — long, tall,
  // thin — which a swept tube cannot be at all, because a sweep is circular in
  // section at every station; built that way the base read as a bundle of pipes.
  // The cost of the anisotropic field is that a cluster is clipped by
  // `(|p/R| - 1)·min R`, whose slack goes as the envelope's own eccentricity, so
  // a fin far longer than it is thin makes the march understep and a grazing ray
  // runs out of its forty-eight iterations before it arrives. That failure is a
  // *hole*, reported nowhere.
  let shot = 0, missed = 0, worst = 0;
  for (const node of leaves(planBonsai(specFor(BONSAI_POND_CANOPY)).nodes)) {
    if (node.kind !== "cluster" || node.field !== "seeded-lobes") continue;
    const lobe = node.lobe;
    worst = Math.max(worst, Math.max(lobe.x, lobe.y, lobe.z) / Math.min(lobe.x, lobe.y, lobe.z));
    const record = {
      kind: "smooth-union-cluster", primitiveId: 1, materialId: 1,
      center_m: { x: 0, y: 0, z: 0 }, lobeRadii_m: lobe, clusterReference: 0,
      packing: {
        field: "seeded-lobes", smoothRadius_m: node.smoothRadius, seed: node.seed >>> 0,
        lobeCount: node.lobeCount, anisotropy: node.anisotropy,
      },
    } as const;
    // Through the envelope's own centre from every direction, so a miss can only
    // be the march giving up: the field is guaranteed to contain its centre.
    for (let i = 0; i < 14; i += 1) {
      for (let j = 0; j < 14; j += 1) {
        const theta = Math.PI * (j + 0.5) / 14;
        const phi = 2 * Math.PI * (i + 0.5) / 14;
        const d = { x: Math.sin(theta) * Math.cos(phi), y: Math.cos(theta), z: Math.sin(theta) * Math.sin(phi) };
        shot += 1;
        if (!intersectSvoPrimitive(record, {
          origin_m: { x: -2 * d.x, y: -2 * d.y, z: -2 * d.z }, direction: d, tMin_m: 0, tMax_m: 4,
        })) missed += 1;
      }
    }
  }
  assert.ok(shot > 0, "the buttresses are not anisotropic fields any more");
  assert.equal(missed, 0, `${missed} of ${shot} rays march through a fin without finding it`);
  // And the other side of the trade: a fin that has regularised back toward a
  // tube is the defect this replaced.
  assert.ok(worst > 3.5, `the fins are only ${worst.toFixed(1)} times longer than thick; they will read as tubes again`);
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
  // And thick where they leave, not just long: a buttress under about four
  // fifths of the stump emerges as a rib on the flare rather than as part of it.
  assert.ok(BONSAI_POND_CANOPY.rootWidth > 0.8, "the buttresses have gone back to being fingers");
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
  assert.ok(fine.leafCount > 0.6 * SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    "one halving should already take most of the whole scene's candidate index");
});

test("the floret's span on a lattice is reported, and is not the constraint", () => {
  // Kept because it is the first thing anyone asks, and the answer is now
  // emphatic: the hero crown's floret is *half a cell* across on a 25 mm
  // lattice, and it still silhouettes exactly, because a scenery primitive's
  // primary visibility is a BVH over exact records and never consults the
  // per-voxel material owner. Under the explicit crown the same specimen's cap
  // spanned two and a third cells — and rendered no better. What the number
  // bounds is how coarsely the cone tracer lights the crown, not what shape it
  // is, which is why an aggregate could go four times finer without asking.
  assert.ok(Math.abs(bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.025) - 0.351) < 0.01);
  assert.ok(Math.abs(bonsaiFloretVoxelSpan(POND_FLORETS, 0.025) - 2.34) < 0.02);
  assert.ok(bonsaiFloretVoxelSpan(BONSAI_SHELF_MINIATURE, 0.025) < 1);
  assert.throws(() => bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0), /positive cell size/);
});
