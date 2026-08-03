import assert from "node:assert/strict";
import test from "node:test";

import { SVO_MATERIAL_FUNCTION_IDS, svoMaterialFunctionIdForEnvironmentProxy } from "../lib/svo-material-abi";
import { bakePondVesselTerrain, pondVesselPlanCurve } from "../lib/voxel-scenery/pond-vessel";
import {
  SWEPT_COPING_POND_BULLNOSE,
  sweptCopingNodes,
  sweptCopingSection,
  type SweptCopingSpec,
} from "../lib/voxel-scenery/swept-coping";
import {
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_TERRAIN_SAMPLE_M,
  HERO_GARDEN_VESSEL,
  HERO_GARDEN_WATERLINE_M,
} from "../lib/hero-garden-scene";
import { terrainHeightAt } from "../lib/terrain";

/**
 * The oracles `lib/voxel-scenery/README.md` asks every generator for:
 * determinism, siblinghood under a re-seed, leaf count against a stated budget,
 * extent, seating on the ground, and the material-closure regex the object's
 * whole surface depends on.
 *
 * Plus the two properties that are the reason this generator exists at all — the
 * section overhangs, and the ground it is set into still contains the water
 * without it.
 */

const RAIL = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
const FLAT_VESSEL = { ...HERO_GARDEN_VESSEL, crest: "flat" as const };
const FLAT_TERRAIN = {
  baseHeight_m: FLAT_VESSEL.groundHeight_m,
  features: [],
  grid: bakePondVesselTerrain(FLAT_VESSEL, HERO_GARDEN_CONTAINER, HERO_GARDEN_TERRAIN_SAMPLE_M),
};
const GROUND = (x_m: number, z_m: number) => terrainHeightAt(FLAT_TERRAIN, x_m, z_m);

function spec(overrides: Partial<SweptCopingSpec> = {}): SweptCopingSpec {
  return {
    ...SWEPT_COPING_POND_BULLNOSE,
    key: "coping", rail: RAIL, groundHeightAt: GROUND,
    material: { palette: "stone", value: 0.92 }, seed: 0x5701e5, ...overrides,
  };
}

test("the section overhangs, which is the whole reason it is not a heightfield", () => {
  const section = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  assert.ok(section.rollBack_m > 0.004,
    "a rim whose widest point is not above the ground has nothing a heightfield cannot do");
  // Crest, footprint and roll-back are one circle and must stay consistent.
  assert.ok(Math.abs(section.radius_m * (1 + SWEPT_COPING_POND_BULLNOSE.undercut)
    - SWEPT_COPING_POND_BULLNOSE.crestHeight_m) < 1e-12);
  assert.ok(Math.abs(section.radius_m - section.groundHalfWidth_m - section.rollBack_m) < 1e-12);
  assert.ok(section.groundHalfWidth_m > 0);

  // A flat-topped section is legal and is the degenerate case: no roll-back, but
  // still a vertical meeting, which a tangent heightfield profile also cannot do.
  assert.equal(sweptCopingSection({ crestHeight_m: 0.05, undercut: 0 }).rollBack_m, 0);
  assert.throws(() => sweptCopingSection({ crestHeight_m: 0.05, undercut: 1 }), RangeError);
  assert.throws(() => sweptCopingSection({ crestHeight_m: 0, undercut: 0.5 }), RangeError);
});

test("one seed grows one coping, and another grows a sibling", () => {
  assert.deepEqual(sweptCopingNodes(spec()), sweptCopingNodes(spec()));

  const first = sweptCopingNodes(spec());
  const sibling = sweptCopingNodes(spec({ seed: 0x5701e5 ^ 0x9e3779b9 }));
  assert.equal(sibling.length, first.length, "a re-seed must not change the record count");
  assert.notDeepEqual(sibling, first);

  // Recognisably the same species: every radius stays inside the modulation the
  // form authorises, so no seed can grow a bead or a thread.
  const base = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  const bound = base.radius_m * (1 + 1.6 * SWEPT_COPING_POND_BULLNOSE.sectionVariation)
    + 2 * SWEPT_COPING_POND_BULLNOSE.relief_m;
  const floor = base.radius_m * (1 - 1.6 * SWEPT_COPING_POND_BULLNOSE.sectionVariation)
    - 2 * SWEPT_COPING_POND_BULLNOSE.relief_m;
  for (const node of sibling) {
    if (node.kind !== "ellipsoid") continue;
    assert.ok(node.radius.x <= bound && node.radius.x >= floor, `radius ${node.radius.x} left the species`);
  }
});

test("records are two per rail segment, and the rail's resolution is the only knob", () => {
  // The budget: 4096 leaves shared by the whole frame, of which the hero set
  // already spends about 1500. Two records per segment is the price of a section
  // that varies (a cone) with no wedge open at the bends (a sphere).
  assert.equal(sweptCopingNodes(spec()).length, 2 * RAIL.length);
  assert.equal(sweptCopingNodes(spec({ segmentStride: 2 })).length, RAIL.length);
  assert.equal(sweptCopingNodes(spec({ rail: pondVesselPlanCurve(HERO_GARDEN_VESSEL, 48) })).length, 2 * 7 * 48);
  assert.ok(2 * 7 * 48 < 4096 - 1600, "the smooth-shading resolution must still leave the hero set its budget");

  assert.throws(() => sweptCopingNodes(spec({ rail: [[0, 0], [1, 0]] })), RangeError);
  assert.throws(() => sweptCopingNodes(spec({ segmentStride: 0 })), RangeError);
  assert.throws(() => sweptCopingNodes(spec({ variationLobes: 2 })), RangeError);
});

test("every node is seated on the ground it was handed, not on a plane", () => {
  const section = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  const nodes = sweptCopingNodes(spec());
  let checked = 0;
  for (const node of nodes) {
    if (node.kind !== "ellipsoid") continue;
    const at = node.place?.position;
    assert.ok(at, "a coping node must be placed");
    const ground = GROUND(at.x, at.z);
    const crest = at.y + node.radius.x - ground;
    // The crest stands proud by the authored height, within the modulation.
    assert.ok(crest > 0.7 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m
      && crest < 1.4 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m, `crest ${crest} at ${at.x},${at.z}`);
    // And the section is buried, so the solid is set into the plaster rather
    // than resting on it: a rim that floats has a seam of sky under it.
    // At the pond form the section's underside sits exactly `undercut * radius`
    // below the plaster, so a quarter of a radius is a bound with room in it
    // rather than the identity restated.
    assert.ok(at.y - node.radius.x < ground - 0.25 * section.radius_m, "the section must be bedded in");
    checked += 1;
  }
  assert.equal(checked, RAIL.length);
});

test("the group is what selects the stone closure, and the regex is load-bearing", () => {
  for (const node of sweptCopingNodes(spec())) {
    assert.equal(
      svoMaterialFunctionIdForEnvironmentProxy({ group: node.group ?? "", tags: node.tags ?? [] }),
      SVO_MATERIAL_FUNCTION_IDS.stone,
      `${node.id} must take the granite surface`,
    );
  }
});

test("omitting the crest from the heightfield does not lower the wall the water rests against", () => {
  // The bet the swept form makes: a coping stands above the still waterline, so
  // spelling it as scenery — which the solver cannot see — costs the water
  // nothing. This pins the freeboard that makes that true.
  const bullnose = {
    baseHeight_m: HERO_GARDEN_VESSEL.groundHeight_m, features: [],
    grid: bakePondVesselTerrain(HERO_GARDEN_VESSEL, HERO_GARDEN_CONTAINER, HERO_GARDEN_TERRAIN_SAMPLE_M),
  };
  let lowestFlat = Infinity, lowestBullnose = Infinity;
  for (const [x, z] of RAIL) {
    const radius = Math.hypot(x, z);
    const outer: readonly [number, number] = [x * (radius + 0.09) / radius, z * (radius + 0.09) / radius];
    lowestFlat = Math.min(lowestFlat, terrainHeightAt(FLAT_TERRAIN, outer[0], outer[1]));
    lowestBullnose = Math.min(lowestBullnose, terrainHeightAt(bullnose, outer[0], outer[1]));
  }
  assert.ok(Math.abs(lowestFlat - lowestBullnose) < 1e-9,
    "the ground outside the coping is the same either way — the crest was never the wall");
  assert.ok(lowestFlat - HERO_GARDEN_WATERLINE_M > 0.03,
    "the terrain alone must hold the still waterline with a cell or more to spare");
});
