import assert from "node:assert/strict";
import test from "node:test";

import {
  HERO_GARDEN_CELL_M,
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_WATER_BELOW_GROUND_M,
  HERO_GARDEN_HOSE_MOUTH_M,
  HERO_GARDEN_HOSE_IMPACT_M,
  HERO_GARDEN_VESSEL,
  HERO_GARDEN_WATERLINE_M,
  createHeroGardenHoseScene,
} from "../lib/hero-garden-scene";
import { validateScene } from "../lib/model";
import { planSceneRuntime } from "../lib/scene-runtime";
import { sampleTerrainGrid, terrainHeightAt, validateTerrain } from "../lib/terrain";
import {
  bakePondVesselTerrain,
  pondVesselCrestProfile,
  pondVesselHeightAt,
  pondVesselPlanCurve,
  pondVesselPlanDistance,
  type PondVesselSpec,
} from "../lib/voxel-scenery/pond-vessel";

/** A plain round vessel, so a profile assertion isolates the profile. */
const PLAIN: PondVesselSpec = {
  center_m: [0, 0],
  radius_m: [0.4, 0.4],
  groundHeight_m: 0.3,
  basinDepth_m: 0.12,
  rimHeight_m: 0.05,
  rimHalfWidth_m: 0.05,
  innerFace_m: 0.15,
  lobes: 8,
  wobble: 0,
  sectionHeightVariation: 0,
  sectionWidthVariation: 0,
  relief_m: 0,
  seed: 1,
};

const CONTAINER = { width_m: 1.2, height_m: 0.6, depth_m: 1.2 };

test("the crest profile is round at the crown and tangent at the foot", () => {
  assert.equal(pondVesselCrestProfile(0), 1);
  assert.equal(pondVesselCrestProfile(1), 0);
  assert.equal(pondVesselCrestProfile(1.4), 0, "beyond the foot the rim contributes nothing");
  // Tangency is what stops the coping meeting the ground in a visible crease,
  // and a circular section — the fullest bullnose — would arrive with slope
  // -infinity instead. The approach is as the square root of the remaining
  // distance, so it is only visible as a limit: each ten-fold narrower window
  // must show a slope smaller by about sqrt(10).
  const slopeOver = (window: number) => Math.abs((pondVesselCrestProfile(1) - pondVesselCrestProfile(1 - window)) / window);
  let previous = Infinity;
  for (const window of [1e-2, 1e-3, 1e-4, 1e-5]) {
    const slope = slopeOver(window);
    assert.ok(slope < previous, `the foot must keep flattening; ${window} gave ${slope}`);
    previous = slope;
  }
  assert.ok(previous < 0.01, `the foot must reach tangency, closest slope was ${previous}`);
  // ...and the crown has to be a dome, not a plateau: the reference's rim is a
  // bullnose. A quartic bump (0.5625 here) flattens the top of it; a true
  // circle (0.8660) is the unreachable ideal with the vertical foot.
  assert.ok(pondVesselCrestProfile(0.5) > 0.62, "the crown must stay full at half-width");
});

test("the plan curve closes, and the wobble moves it without pinching the rim", () => {
  const round = pondVesselPlanCurve(PLAIN);
  assert.ok(round.length >= 3 * 16);
  for (const [x, z] of round) {
    // Exactly the ellipse, not nearly: the polar construction exists so that
    // `radius_m` is the radius. A Cartesian spline through the same control
    // points scallops inward by several millimetres between them.
    assert.ok(Math.abs(Math.hypot(x, z) - 0.4) < 1e-12, "an unwobbled plan is the authored ellipse");
  }
  const wobbled = pondVesselPlanCurve({ ...PLAIN, wobble: 0.12, seed: 7 });
  const radii = wobbled.map(([x, z]) => Math.hypot(x, z));
  assert.ok(Math.max(...radii) - Math.min(...radii) > 0.02, "the seed must actually move the outline");
});

test("with the crest swept as a solid, the plaster runs level to the plan curve and nothing else moved", () => {
  // This test used to be "the coping's section varies as it runs", measured on
  // `pondVesselHeightAt`, and it was the right assertion until the crest left
  // the heightfield. `HERO_GARDEN_VESSEL.crest` is `"flat"` now: the swelling it
  // measured is a swept solid (`lib/voxel-scenery/swept-coping.ts`), this
  // function's crest term is identically zero on the hero vessel, and the
  // section-varies property went with it — it is asserted on the solid's own run
  // in `tests/swept-coping.test.ts`.
  //
  // What is left here is the vessel's half of that contract, which nothing was
  // asserting: the plaster the solid is set into. Both claims below are the ones
  // the doc comment on `crest` makes, and they are the reason omitting the crest
  // is a rendering decision rather than a containment one.
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
  const bullnose: PondVesselSpec = { ...HERO_GARDEN_VESSEL, crest: "bullnose" };

  // One: level to the plan curve. The ground on the centreline is
  // `groundHeight_m` plus the relief and nothing else, which is what gives the
  // swept rim a flat band to be bedded into rather than a shoulder to perch on.
  // Measured, the plaster under the rail wanders by -2.24 to +2.03 mm, which is
  // the 2.5 mm relief and none of it a crest.
  for (const [x, z] of curve) {
    const height = pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z);
    assert.ok(Math.abs(height - HERO_GARDEN_VESSEL.groundHeight_m) <= HERO_GARDEN_VESSEL.relief_m,
      `the plaster at (${x.toFixed(3)}, ${z.toFixed(3)}) stands ${((height - HERO_GARDEN_VESSEL.groundHeight_m) * 1e3).toFixed(2)} mm off level`);
    // ...and what left was a crest at its authored height, not a shaving off the
    // ground. Measured 46.2 to 63.9 mm against the authored 55, which is the
    // section's own height modulation and nothing else.
    const crest = pondVesselHeightAt(bullnose, curve, x, z) - height;
    assert.ok(crest > 0.6 * HERO_GARDEN_VESSEL.rimHeight_m && crest < 1.4 * HERO_GARDEN_VESSEL.rimHeight_m,
      `the crest that left the heightfield was ${(crest * 1e3).toFixed(2)} mm, not the authored ${HERO_GARDEN_VESSEL.rimHeight_m * 1e3} mm`);
  }

  // Two: it lowered nothing the water can reach. The doc comment claims the
  // still waterline moves "by less than a nanometre"; measured it moves by
  // exactly zero, and this is why. `pondVesselCrestProfile` is identically zero
  // past the coping's foot, and the foot never runs wider than
  // `rimHalfWidth_m * (1 + 1.6 * sectionWidthVariation)` — 42.9 mm, the widest
  // a Catmull-Rom through values in [-1, 1] can overshoot to. Outside that the
  // two vessels are the same float, so the basin floor, the inner face and every
  // metre of plaster the water rests against are untouched by the decision.
  const widestBand_m = HERO_GARDEN_VESSEL.rimHalfWidth_m * (1 + 1.6 * HERO_GARDEN_VESSEL.sectionWidthVariation);
  let compared = 0;
  for (let x = -0.9; x <= 0.9; x += 0.015) for (let z = -0.6; z <= 0.6; z += 0.015) {
    if (Math.abs(pondVesselPlanDistance(curve, x, z)) <= widestBand_m) continue;
    assert.equal(pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z), pondVesselHeightAt(bullnose, curve, x, z),
      `omitting the crest moved the ground at (${x.toFixed(3)}, ${z.toFixed(3)}), which is outside the widest band the crest can reach`);
    compared += 1;
  }
  assert.ok(compared > 5000, `only ${compared} samples fell outside the band; the sweep must reach the basin and the ground beyond the pond`);
});

test("a tank fill wets the basin and nothing else", () => {
  // The property the scene failed to load without. `tank-fill` wets every column
  // the waterline clears, so ground outside the coping that sits below it floods
  // the whole container with a film two cells deep — a degenerate topology the
  // sparse authority rejects outright, and not what the reference shows either.
  const scene = createHeroGardenHoseScene();
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
  const clearance_m: number[] = [];
  for (let x = -0.9; x <= 0.9; x += 0.01) for (let z = -0.6; z <= 0.6; z += 0.01) {
    // Outside the coping's outer foot, with a cell of margin.
    if (pondVesselPlanDistance(curve, x, z) < 1.5 * HERO_GARDEN_VESSEL.rimHalfWidth_m) continue;
    clearance_m.push(terrainHeightAt(scene.terrain, x, z) - HERO_GARDEN_WATERLINE_M);
  }
  assert.ok(clearance_m.length > 1000, "the sweep must actually reach the ground outside the pond");
  const worst = Math.min(...clearance_m);
  assert.ok(worst > 0, `ground outside the pond dips ${(worst * 1e3).toFixed(1)} mm under the waterline`);
  // And by more than a cell, so the fill cannot wet it through rounding.
  assert.ok(worst > HERO_GARDEN_CELL_M, `only ${(worst * 1e3).toFixed(1)} mm of clearance, under one cell`);
});

test("the plan distance is signed, and zero on the curve", () => {
  const curve = pondVesselPlanCurve(PLAIN);
  assert.ok(pondVesselPlanDistance(curve, 0, 0) < -0.39, "the centre is deep inside");
  assert.ok(pondVesselPlanDistance(curve, 0.9, 0) > 0.49, "well outside is positive");
  for (const [x, z] of curve) {
    assert.ok(Math.abs(pondVesselPlanDistance(curve, x, z)) < 1e-9, "a point on the curve is on the curve");
  }
});

test("the vessel is a basin: floor below ground, rim above it, monotone between", () => {
  const curve = pondVesselPlanCurve(PLAIN);
  const at = (x: number) => pondVesselHeightAt(PLAIN, curve, x, 0);
  const floor = PLAIN.groundHeight_m - PLAIN.basinDepth_m;
  assert.ok(Math.abs(at(0) - floor) < 1e-9, "the middle of the basin is at the floor");
  assert.ok(Math.abs(at(0.4) - (PLAIN.groundHeight_m + PLAIN.rimHeight_m)) < 1e-9, "the crest is on the curve");
  assert.ok(Math.abs(at(0.9) - PLAIN.groundHeight_m) < 1e-9, "far outside is plain ground");
  // Climbing the inner face must never turn back down, or the water finds a
  // ledge to pool on that the outline says nothing about.
  let previous = -Infinity;
  for (let x = 0; x <= 0.4; x += 0.005) {
    const height = at(x);
    assert.ok(height >= previous - 1e-9, `inner face reverses at x=${x}`);
    previous = height;
  }
});

test("the rim is closed above the waterline all the way round", () => {
  // The one structural property the water depends on. Sampled on the crest
  // centreline rather than on the lattice, so this is the vessel's own claim;
  // the bake's version of it is asserted separately below.
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
  for (const [x, z] of curve) {
    const height = pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z);
    assert.ok(height >= HERO_GARDEN_WATERLINE_M + HERO_GARDEN_WATER_BELOW_GROUND_M - 1e-6,
      `rim at (${x}, ${z}) stands only ${(height - HERO_GARDEN_WATERLINE_M).toFixed(4)} m above the waterline`);
  }
});

test("a terrace raises ground outside the pond and never intrudes on the rim", () => {
  const spec: PondVesselSpec = { ...PLAIN, terraces: [{ center_m: [0.75, 0], radius_m: [0.35, 0.3], height_m: 0.08, flat: 0.3 }] };
  const curve = pondVesselPlanCurve(spec);
  assert.ok(pondVesselHeightAt(spec, curve, 0.75, 0) > PLAIN.groundHeight_m + 0.07, "the terrace rises");
  // Its mask is zero across the coping, so a plateau can be authored right up
  // against the pond without swallowing the rim it abuts.
  const onCrest = pondVesselHeightAt(spec, curve, 0.4, 0);
  assert.ok(Math.abs(onCrest - (PLAIN.groundHeight_m + PLAIN.rimHeight_m)) < 1e-9, "the crest is untouched");
  assert.ok(pondVesselHeightAt(spec, curve, 0, 0) < PLAIN.groundHeight_m, "the basin is untouched");
});

test("the bake reproduces the generator on the lattice the solver reads", () => {
  const grid = bakePondVesselTerrain(PLAIN, CONTAINER, 0.0075);
  const curve = pondVesselPlanCurve(PLAIN);
  assert.deepEqual(validateTerrain({ baseHeight_m: PLAIN.groundHeight_m, features: [], grid }, CONTAINER), []);
  // Sampling *between* nodes is the honest check: it is what every solver does,
  // and it is where a bake that disagreed with its generator would show up.
  let worst = 0;
  for (let x = -0.55; x <= 0.55; x += 0.017) for (let z = -0.55; z <= 0.55; z += 0.017) {
    worst = Math.max(worst, Math.abs(sampleTerrainGrid(grid, x, z) - pondVesselHeightAt(PLAIN, curve, x, z)));
  }
  // Bilinear across a 7.5 mm cell. The error concentrates at the coping's foot,
  // where the crest profile reaches tangency with unbounded curvature — the
  // crown, the inner face and the open ground are all an order of magnitude
  // better. Two millimetres is that interpolation error, not a disagreement
  // between the bake and the generator.
  assert.ok(worst < 2e-3, `bake departs from the generator by ${(worst * 1e3).toFixed(2)} mm`);
});

test("water is one flag, and turning it on changes nothing else", () => {
  const dry = createHeroGardenHoseScene();
  const wet = createHeroGardenHoseScene({ water: true });
  assert.deepEqual(validateScene(dry), []);
  assert.deepEqual(validateScene(wet), []);

  // The point of the switch. If the two documents could differ anywhere else,
  // the dry scene would be a second scene wearing the first one's name, and
  // feedback given on it would be feedback on something that is not the hero.
  assert.equal(planSceneRuntime(dry).fluidSolver, false, "the scene opens without a solver");
  assert.equal(planSceneRuntime(wet).fluidSolver, true);
  assert.deepEqual(
    { ...dry, systems: undefined },
    { ...wet, systems: undefined },
    "dry and wet must be the same document apart from systems.fluid",
  );
  assert.deepEqual(dry.systems, { ...wet.systems, fluid: false });

  // Including the water itself: the fill and the jet are authored in both, so
  // turning it on fills the pond that was designed rather than an empty tank.
  assert.ok(dry.container.fillFraction > 0 && dry.fluid.inflow, "the dry document still describes its water");
});

test("the hero document is valid, deterministic, and holds its water", () => {
  const scene = createHeroGardenHoseScene({ water: true });
  assert.deepEqual(validateScene(scene), []);
  assert.equal(JSON.stringify(scene), JSON.stringify(createHeroGardenHoseScene({ water: true })), "the bake must be deterministic");

  // Every container dimension is a whole number of bricks, so the sparse domain
  // has no partial brick at a wall.
  const brick_m = HERO_GARDEN_CELL_M * 8;
  for (const size of [HERO_GARDEN_CONTAINER.width_m, HERO_GARDEN_CONTAINER.height_m, HERO_GARDEN_CONTAINER.depth_m]) {
    assert.ok(Math.abs(size / brick_m - Math.round(size / brick_m)) < 1e-9, `${size} m is not a whole number of ${brick_m} m bricks`);
  }

  // The initial fill is the waterline, and the waterline clears the basin floor
  // by enough water to be a pond rather than a puddle.
  assert.ok(Math.abs(scene.container.fillFraction * HERO_GARDEN_CONTAINER.height_m - HERO_GARDEN_WATERLINE_M) < 1e-9);
  const floor = HERO_GARDEN_VESSEL.groundHeight_m - HERO_GARDEN_VESSEL.basinDepth_m;
  assert.ok(HERO_GARDEN_WATERLINE_M - floor > 0.1, "the pond should be over 10 cm deep at the middle");

  // And the hose is aimed into the water, not at the coping.
  const inflow = scene.fluid.inflow!;
  const groundUnderJet = terrainHeightAt(scene.terrain, HERO_GARDEN_HOSE_IMPACT_M.x, HERO_GARDEN_HOSE_IMPACT_M.z);
  assert.ok(groundUnderJet < HERO_GARDEN_WATERLINE_M, "the jet must land inside the basin");
  const impactTime = (HERO_GARDEN_HOSE_IMPACT_M.y - inflow.center_m.y) / inflow.velocity_m_s.y;
  const landing = {
    x: inflow.center_m.x + impactTime * inflow.velocity_m_s.x,
    z: inflow.center_m.z + impactTime * inflow.velocity_m_s.z,
  };
  assert.ok(Math.hypot(landing.x - HERO_GARDEN_HOSE_IMPACT_M.x, landing.z - HERO_GARDEN_HOSE_IMPACT_M.z) < 1e-9,
    "the authored jet direction must reach its visible impact point");
  assert.ok(inflow.center_m.y > HERO_GARDEN_VESSEL.groundHeight_m + HERO_GARDEN_VESSEL.rimHeight_m,
    "the hose mouth must clear the rim it arrives over");
});

test("the drawn hose and the injected jet leave from the same place, pointing the same way", () => {
  // The most obvious way this scene could look wrong is a nozzle that is not
  // where the water appears. Both are derived from one mouth and one aim, and
  // this is the assertion that keeps them that way.
  const scene = createHeroGardenHoseScene();
  const inflow = scene.fluid.inflow!;
  assert.deepEqual(inflow.center_m, { ...HERO_GARDEN_HOSE_MOUTH_M });

  const ferrule = scene.scenery!.nodes.find((node) => node.id === "hose/ferrule");
  assert.ok(ferrule && ferrule.kind === "capsule", "the nozzle collar must exist");
  assert.deepEqual(ferrule.from, { ...HERO_GARDEN_HOSE_MOUTH_M });

  // The collar runs back up the hose, so it points into the jet: the angle
  // between the water's velocity and the tube's own axis has to be a straight one.
  const axis = { x: ferrule.to.x - ferrule.from.x, y: ferrule.to.y - ferrule.from.y, z: ferrule.to.z - ferrule.from.z };
  const dot = axis.x * inflow.velocity_m_s.x + axis.y * inflow.velocity_m_s.y + axis.z * inflow.velocity_m_s.z;
  const cosine = dot / (Math.hypot(axis.x, axis.y, axis.z) * Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z));
  assert.ok(cosine < -0.999, `nozzle and jet disagree on direction, cosine ${cosine}`);

  // The run is continuous. The hose is a family of tapered-sweep records rather
  // than a capsule chain — one record carries several bends with a smooth
  // minimum through them — so continuity is no longer "this capsule's `to` is
  // the next one's `from`". Each record places its control points in its own
  // fitted frame, and the property that still has to hold is that consecutive
  // records *share a station in world space*: the fit is allowed to choose where
  // to cut the path, but not to leave a gap in it.
  const runs = scene.scenery!.nodes.filter((node) => node.id.startsWith("hose/tube"));
  assert.ok(runs.length >= 1, "the hose must actually run somewhere");
  const worldPoints = runs.map((node) => {
    assert.equal(node.kind, "cluster");
    const sweep = node as Extract<typeof node, { kind: "cluster" }>;
    assert.equal("field" in sweep ? sweep.field : undefined, "tapered-sweep");
    assert.equal(sweep.place?.units, "metres", "the hose is authored in metres, not scene fractions");
    const origin = sweep.place!.position!;
    const q = sweep.place!.orientation!;
    assert.ok(Math.abs(Math.hypot(q.w, q.x, q.y, q.z) - 1) < 1e-9, "a record's frame must be a rotation");
    return (sweep as { points: readonly { position: { x: number; y: number; z: number }; radius: number }[] })
      .points.map(({ position: p, radius }) => {
        assert.ok(radius > 0, "every station carries a bore");
        // The standard quaternion sandwich, spelled out: the expander rotates a
        // cluster's control points by the node's frame and offsets them by its
        // origin, and this has to agree with it to be checking anything.
        const t = { x: 2 * (q.y * p.z - q.z * p.y), y: 2 * (q.z * p.x - q.x * p.z), z: 2 * (q.x * p.y - q.y * p.x) };
        return {
          x: origin.x + p.x + q.w * t.x + (q.y * t.z - q.z * t.y),
          y: origin.y + p.y + q.w * t.y + (q.z * t.x - q.x * t.z),
          z: origin.z + p.z + q.w * t.z + (q.x * t.y - q.y * t.x),
        };
      });
  });
  for (let index = 1; index < worldPoints.length; index += 1) {
    const end = worldPoints[index - 1][worldPoints[index - 1].length - 1];
    const start = worldPoints[index][0];
    assert.ok(Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z) < 1e-9,
      `the hose breaks between records ${index - 1} and ${index}`);
  }
  // And it starts at the mouth the jet leaves from, which is the whole point of
  // the test above holding for the collar as well.
  const head = worldPoints[0][0];
  assert.ok(Math.hypot(head.x - HERO_GARDEN_HOSE_MOUTH_M.x, head.y - HERO_GARDEN_HOSE_MOUTH_M.y,
    head.z - HERO_GARDEN_HOSE_MOUTH_M.z) < 1e-9, "the tube must start at the nozzle");
});
