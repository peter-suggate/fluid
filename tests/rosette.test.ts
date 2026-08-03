import assert from "node:assert/strict";
import test from "node:test";

import { HERO_GARDEN_VESSEL } from "../lib/hero-garden-scene";
import { validateSceneryGraph, walkSceneryNodes, type SceneryGraph, type SceneryNode } from "../lib/scenery-graph";
import { pondVesselHeightAt, pondVesselPlanCurve } from "../lib/voxel-scenery/pond-vessel";
import {
  planRosette,
  rosetteBladeHalfWidth_m,
  rosetteNodes,
  ROSETTE_AIR_PLANT,
  ROSETTE_GRASS_TUFT,
  type RosetteForm,
  type RosettePlan,
  type RosetteSpec,
} from "../lib/voxel-scenery/rosette";

/** A level lawn, so an extent assertion measures the plant and not the bank. */
const FLAT_GROUND_M = 0.3;
const flat = () => FLAT_GROUND_M;

function spec(form: RosetteForm, overrides: Partial<RosetteSpec> = {}): RosetteSpec {
  return { ...form, key: "plant", at_m: [0, 0], groundHeightAt: flat, seed: 0x1234, ...overrides };
}

/** Every primitive a plan publishes, flattened out of its one group node. */
function primitives(nodes: readonly SceneryNode[]): SceneryNode[] {
  return [...walkSceneryNodes(nodes)].map(({ node }) => node).filter((node) => node.kind !== "group");
}

/** Tip of every blade, relative to the plant's root. */
function tips(plan: RosettePlan): { x: number; y: number; z: number }[] {
  return plan.blades.map((blade) => {
    const last = blade.segments[blade.segments.length - 1].to_m;
    return { x: last.x - plan.root_m.x, y: last.y - plan.root_m.y, z: last.z - plan.root_m.z };
  });
}

test("both forms are the size the reference's plants are", () => {
  // Measured off `garden-pond-hose-fill-simplified.png` with a pebble in the
  // same plane for a ruler: the right-hand air plant is about 80 mm across and
  // 67 mm tall, the near-left tuft about 90 by 85. The absolute bands are loose
  // because a photographic ruler is worth a few millimetres at best, and the
  // aspect assertion below is the one that is actually tight.
  const air = planRosette(spec(ROSETTE_AIR_PLANT));
  const tuft = planRosette(spec(ROSETTE_GRASS_TUFT));
  assert.ok(2 * air.radius_m > 0.065 && 2 * air.radius_m < 0.095, `air plant is ${(2000 * air.radius_m).toFixed(0)} mm across`);
  assert.ok(air.height_m > 0.055 && air.height_m < 0.080, `air plant is ${(1000 * air.height_m).toFixed(0)} mm tall`);
  assert.ok(2 * tuft.radius_m > 0.075 && 2 * tuft.radius_m < 0.110, `tuft is ${(2000 * tuft.radius_m).toFixed(0)} mm across`);
  assert.ok(tuft.height_m > 0.070 && tuft.height_m < 0.100, `tuft is ${(1000 * tuft.height_m).toFixed(0)} mm tall`);
  // The air plant is wider than it is tall — a rosette whose oldest blades
  // recurve past horizontal has to be — and it opens further for its height
  // than the tuft, whose needles cannot splay as far without reading as a
  // shuttlecock. That relationship, not either size on its own, is what
  // separates the two silhouettes at a hundred pixels.
  assert.ok(2 * air.radius_m > air.height_m, "the air plant must be wider than it is tall");
  assert.ok(air.radius_m / air.height_m > 1.08 * (tuft.radius_m / tuft.height_m),
    "the air plant must open clearly wider for its height than the tuft");
});

test("a rosette costs the segments it says it costs", () => {
  // The scene's whole primitive budget is 4 096 leaves and the planting's share
  // is 400. A generator whose cost cannot be read off its spec is one that
  // silently spends someone else's share.
  for (const form of [ROSETTE_AIR_PLANT, ROSETTE_GRASS_TUFT]) {
    const plan = planRosette(spec(form));
    assert.equal(plan.leafCount, form.bladeCount * form.segments + 1);
    assert.equal(primitives(rosetteNodes(spec(form))).length, plan.leafCount);
  }
  const set = 3 * planRosette(spec(ROSETTE_AIR_PLANT)).leafCount + planRosette(spec(ROSETTE_GRASS_TUFT)).leafCount;
  assert.ok(set <= 400, `the reference's planting costs ${set} leaves`);
});

test("blade count is the spec's, and every blade is a chain of that many cones", () => {
  const form: RosetteForm = { ...ROSETTE_AIR_PLANT, bladeCount: 11, segments: 4 };
  const plan = planRosette(spec(form));
  assert.equal(plan.blades.length, 11);
  for (const blade of plan.blades) {
    assert.equal(blade.segments.length, 4);
    // Chained: each run starts where the last one ended, so a blade is one
    // curve and not four floating cones.
    for (let index = 1; index < blade.segments.length; index += 1) {
      assert.deepEqual(blade.segments[index].from_m, blade.segments[index - 1].to_m);
    }
  }
});

test("a blade is widest near its base and tapers to a fine tip", () => {
  const halfWidth = ROSETTE_AIR_PLANT.halfWidth_m;
  const samples = Array.from({ length: 51 }, (_, index) => rosetteBladeHalfWidth_m(index / 50, halfWidth));
  const widest = Math.max(...samples);
  assert.ok(Math.abs(widest - halfWidth) < 1e-9, "the widest point is the authored half-width");
  assert.ok(samples.indexOf(widest) / 50 < 0.35, "the shoulder is in the blade's first third");
  assert.ok(samples[0] < 0.7 * halfWidth, "the blade narrows where it leaves the neck");
  // A tip that closes to nothing has no normal to shade. It is a surface, and
  // it is a fifth of a millimetre on a 9 mm blade.
  assert.ok(samples[50] > 0 && samples[50] < 0.1 * halfWidth, "the tip is fine but not degenerate");
  for (let index = 20; index < 50; index += 1) {
    assert.ok(samples[index + 1] <= samples[index], `the outer run never widens (at s=${(index + 1) / 50})`);
  }
});

test("blades recurve: a blade turns most in its outer half, and the oldest fall past horizontal", () => {
  const plan = planRosette(spec(ROSETTE_AIR_PLANT));
  const tilt = (blade: RosettePlan["blades"][number], index: number) => {
    const run = blade.segments[index];
    const axis = { x: run.to_m.x - run.from_m.x, y: run.to_m.y - run.from_m.y, z: run.to_m.z - run.from_m.z };
    return Math.acos(axis.y / Math.hypot(axis.x, axis.y, axis.z));
  };
  for (const blade of plan.blades) {
    const last = blade.segments.length - 1;
    assert.ok(tilt(blade, last) > tilt(blade, 0), "a blade never straightens as it runs out");
    // The turn accumulates outward. Sharing it evenly would make the blade a
    // circular arc, which reads as a hoop rather than as a recurve.
    const inner = tilt(blade, Math.floor(last / 2)) - tilt(blade, 0);
    const outer = tilt(blade, last) - tilt(blade, Math.floor(last / 2));
    assert.ok(outer >= inner, "the outer half of a blade turns at least as far as the inner half");
  }
  // The reference's air plants have blades whose tips point downward. Without
  // one, a rosette reads as a shuttlecock.
  assert.ok(plan.blades.some((blade) => blade.tipTilt_rad > Math.PI / 2),
    "at least one blade recurves past horizontal");
  assert.ok(plan.blades.some((blade) => blade.tipTilt_rad < 0.25 * ROSETTE_AIR_PLANT.splay_rad),
    "at least one blade stays near upright");
});

test("a rosette is a spiral, not a fan: no two blades share a bearing", () => {
  // Nine cones on one azimuth is the failure this pins. The golden angle is the
  // whole reason a nine-blade rosette reads as grown from every direction, and
  // the jitter must not be large enough to close a gap in it.
  const plan = planRosette(spec(ROSETTE_AIR_PLANT));
  const bearings = plan.blades.map((blade) => ((blade.azimuth_rad % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)).sort((a, b) => a - b);
  for (let index = 1; index < bearings.length; index += 1) {
    assert.ok(bearings[index] - bearings[index - 1] > 0.20,
      `blades ${index - 1} and ${index} are ${(bearings[index] - bearings[index - 1]).toFixed(3)} rad apart`);
  }
});

test("the same seed grows the same plant, and a different one grows a different plant", () => {
  const at = spec(ROSETTE_AIR_PLANT);
  assert.deepEqual(rosetteNodes(at), rosetteNodes(spec(ROSETTE_AIR_PLANT)));
  assert.deepEqual(planRosette(at).blades, planRosette(spec(ROSETTE_AIR_PLANT)).blades);
  const other = planRosette(spec(ROSETTE_AIR_PLANT, { seed: 0x1235 }));
  assert.notDeepEqual(planRosette(at).blades, other.blades);
  // A re-seed must still be the same plant. Sibling, not stranger.
  assert.equal(other.leafCount, planRosette(at).leafCount);
  assert.ok(Math.abs(other.radius_m - planRosette(at).radius_m) < 0.02, "a re-seeded plant keeps its size");
});

test("every rosette is seated on the ground it was given", () => {
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
  const groundHeightAt = (x: number, z: number) => pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z);
  // Right round the bank, a little outside the crest, which is where all three
  // of the reference's plants sit.
  for (let step = 0; step < 24; step += 1) {
    const angle = (2 * Math.PI * step) / 24;
    const reach = 1 + 0.05 / Math.hypot(0.52 * Math.cos(angle), 0.38 * Math.sin(angle));
    const at_m = [0.52 * Math.cos(angle) * reach, 0.38 * Math.sin(angle) * reach] as const;
    const plan = planRosette(spec(ROSETTE_AIR_PLANT, { at_m, groundHeightAt, seed: 17 + step }));
    const ground = groundHeightAt(at_m[0], at_m[1]);
    // Wedged, not stood on and not swallowed. Most of the neck is under the
    // surface — that is what "bedded into the stonework" means, and it is what
    // stops a rosette reading as a cut plant laid on the coping — but its crown
    // still clears, so the blades leave a stem and not bare ground.
    assert.ok(plan.root_m.y < ground, "the root is pushed under the surface");
    const neckTop = plan.neckCenter_m.y + plan.neckRadius_m.y;
    const neckBottom = plan.neckCenter_m.y - plan.neckRadius_m.y;
    assert.ok(neckTop > ground, `the neck clears the ground at angle ${angle.toFixed(2)}`);
    const buried = (ground - neckBottom) / (neckTop - neckBottom);
    assert.ok(buried > 0.4 && buried < 0.9, `the neck is ${(100 * buried).toFixed(0)}% buried`);
    assert.ok(neckTop - ground < 0.25 * plan.height_m, "the neck is a stub, not a trunk");
    // Blades leave from inside the neck, so no blade base hangs in the air.
    for (const blade of plan.blades) {
      const base = blade.segments[0].from_m;
      assert.ok(base.y < neckTop, "a blade leaves from inside the neck");
      assert.ok(Math.hypot(base.x - plan.root_m.x, base.z - plan.root_m.z) <= plan.neckRadius_m.x,
        "a blade leaves from inside the neck's own bundle");
    }
    // The plant stands up. Every tip above the ground it grew from is the only
    // way a bank plant can be legible against the coping behind it.
    for (const tip of tips(plan)) {
      assert.ok(plan.root_m.y + tip.y > ground, "no blade tip is driven into the ground");
    }
  }
});

test("bedding is the caller's, and a rosette on a slope follows it", () => {
  const slope = (x: number) => 0.3 + 0.4 * x;
  const plan = planRosette(spec(ROSETTE_AIR_PLANT, { at_m: [0.25, 0], groundHeightAt: (x) => slope(x), bed_m: 0.004 }));
  assert.ok(Math.abs(plan.root_m.y - (slope(0.25) - 0.004)) < 1e-12);
  // A rosette does not tilt with the ground. Neither does any of the
  // reference's, which are wedged into gaps rather than rooted in a bank.
  assert.equal(plan.neckCenter_m.x, plan.root_m.x);
  assert.equal(plan.neckCenter_m.z, plan.root_m.z);
});

test("the published nodes are a valid scenery graph on the hero scene's palettes", () => {
  const nodes = [
    ...rosetteNodes(spec(ROSETTE_AIR_PLANT, { key: "planting/air-0", at_m: [0.4, 0.2] })),
    ...rosetteNodes(spec(ROSETTE_GRASS_TUFT, { key: "planting/tuft-0", at_m: [-0.4, 0.1], seed: 9 })),
  ];
  const graph: SceneryGraph = {
    palettes: { clay: { tint: [1, 0.985, 0.955] }, stone: { tint: [0.972, 0.984, 1] } },
    nodes: [{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" }, ...nodes],
  };
  assert.deepEqual(validateSceneryGraph(graph), []);
  // One group per plant, so a click lands on the plant and not on a blade.
  assert.equal(nodes.length, 2);
  assert.ok(nodes.every((node) => node.kind === "group"));
  for (const node of primitives(nodes)) {
    assert.ok(node.kind === "cone" || node.kind === "ellipsoid", `unexpected ${node.kind}`);
    assert.ok("material" in node && "palette" in node.material && node.material.palette === "clay");
    // The declared band. Outside it the plant is either grey or blown out, and
    // the whole set is authored to read form through value alone.
    const value = "material" in node && "value" in node.material ? node.material.value : NaN;
    assert.ok(value >= 0.62 && value <= 0.92, `${node.id} has palette value ${value}`);
    // `roughnessFor` in the proxy builder keys the matte foliage finish off the
    // group name. A plant in a group it does not match renders as wet plastic.
    assert.ok(/leaf/.test(node.group ?? ""), `${node.id} is in group ${node.group}`);
  }
});

test("the generator refuses a plant it cannot grow", () => {
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, bladeCount: 2 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, bladeCount: 8.5 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, segments: 0 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, halfWidth_m: 0 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, length_m: -0.05 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, launchShare: 1.4 })), RangeError);
  assert.throws(() => planRosette(spec({ ...ROSETTE_AIR_PLANT, recurve: 0 })), RangeError);
});
