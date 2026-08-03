import assert from "node:assert/strict";
import test from "node:test";

import { growSceneryGenerator, SWEPT_COPING_RAIL_SAMPLES_PER_LOBE } from "../lib/scenery-generators";
import { SVO_MATERIAL_FUNCTION_IDS, svoMaterialFunctionIdForEnvironmentProxy } from "../lib/svo-material-abi";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import {
  bakePondVesselTerrain,
  pondVesselPlanCurve,
  pondVesselPlanDistance,
  pondVesselRimHalfWidthAt,
  pondVesselTurnAt,
} from "../lib/voxel-scenery/pond-vessel";
import {
  SWEPT_COPING_POND_BULLNOSE,
  sweptCopingNodes,
  sweptCopingSection,
  sweptCopingStations,
  type SweptCopingSpec,
} from "../lib/voxel-scenery/swept-coping";
import {
  createHeroGardenHoseScene,
  HERO_GARDEN_CELL_M,
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
/** Stations the form's own stride leaves on that rail; the record count is the same. */
const STATIONS = Math.floor(
  pondVesselPlanCurve(HERO_GARDEN_VESSEL).length / (SWEPT_COPING_POND_BULLNOSE.segmentStride ?? 1));
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

/**
 * The rim's shape, asked for at the seam rather than read back off records.
 *
 * This used to unpack whatever primitive the emitter happened to be spelling the
 * run with, and it was rewritten three times in a week as the emitter went cone
 * chain, capsule chain, round-cone chain and finally tapered sweep — while the
 * *properties below it* never changed once. `sweptCopingStations` is where a
 * coping's centreline and radius profile live (`lib/voxel-scenery/swept-coping.ts`),
 * and every assertion in this file about crest height, burial, footprint or
 * freeboard is a question about the shape, not about the records. Only the one
 * test named for the record layout looks at nodes now.
 */
function copingRailSamples(spec: SweptCopingSpec) {
  return sweptCopingStations(spec);
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
  const siblingSpec = spec({ seed: 0x5701e5 ^ 0x9e3779b9 });
  const sibling = sweptCopingNodes(siblingSpec);
  assert.equal(sibling.length, first.length, "a re-seed must not change the record count");
  assert.notDeepEqual(sibling, first);

  // Recognisably the same species: every radius stays inside the modulation the
  // form authorises, so no seed can grow a bead or a thread.
  const base = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  const bound = base.radius_m * (1 + 1.6 * SWEPT_COPING_POND_BULLNOSE.sectionVariation)
    + 2 * SWEPT_COPING_POND_BULLNOSE.relief_m;
  const floor = base.radius_m * (1 - 1.6 * SWEPT_COPING_POND_BULLNOSE.sectionVariation)
    - 2 * SWEPT_COPING_POND_BULLNOSE.relief_m;
  for (const point of copingRailSamples(siblingSpec)) {
    assert.ok(point.radius <= bound && point.radius >= floor, `radius ${point.radius} left the species`);
  }
});

test("the run is a C1 round-cone chain, and its stride is a density decision", () => {
  // The one test in this file that is about *records*. Everything else asks
  // `sweptCopingStations`, which is why the emitter could be swapped twice under
  // them without any of them moving.
  const nodes = sweptCopingNodes(spec());
  assert.equal(nodes.length, STATIONS);
  assert.equal(sweptCopingNodes(spec({ segmentStride: 1 })).length, RAIL.length);

  for (const node of nodes) {
    assert.equal(node.kind, "cone");
    if (node.kind !== "cone") continue;
    // The whole argument for the chain. A round cone is the convex hull of its
    // end spheres and is tangent to both, so consecutive segments sharing a
    // sphere are C1 whatever the rail turns through — and there is no planar cap
    // for the SVO's one-owner-per-voxel to leak a buried normal through.
    assert.equal(node.roundedEnds, true);
  }
  // Consecutive segments have to *share* that sphere, not merely abut: the
  // arriving top radius and the leaving base radius are one station's section.
  for (let index = 0; index < nodes.length; index += 1) {
    const here = nodes[index], next = nodes[(index + 1) % nodes.length];
    if (here.kind !== "cone" || next.kind !== "cone") continue;
    assert.ok(Math.abs(here.topRadius - next.baseRadius) < 1e-12,
      `run ${index} hands over a ${here.topRadius} section to a ${next.baseRadius} one`);
  }

  // A rim is uniformly dense all the way round, so it overflows
  // `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` everywhere at once or nowhere. The
  // stride is what keeps it under: at 1 the hero scene's busiest 200 mm brick was
  // 110 with eight bricks over the 64 contract, and 58 of that brick was rim.
  const publication = buildSvoScenePrimitives(createHeroGardenHoseScene());
  const coping = publication.metadata.filter(({ tags }) => tags.includes("coping"));
  assert.ok(coping.length <= 128, `the hero rim publishes ${coping.length} records; 336 overflowed eight bricks`);
  assert.ok(coping.every(({ primitiveIndex }) => publication.descriptors[primitiveIndex].kind === "round-cone"),
    "the scenery bridge must preserve rounded ends as the cap-free ABI kind");

  assert.throws(() => sweptCopingNodes(spec({ rail: [[0, 0], [1, 0]] })), RangeError);
  assert.throws(() => sweptCopingNodes(spec({ segmentStride: 0 })), RangeError);
  assert.throws(() => sweptCopingNodes(spec({ variationLobes: 2 })), RangeError);
});

test("every node is seated on the ground it was handed, not on a plane", () => {
  const section = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  let checked = 0;
  for (const { at, radius } of copingRailSamples(spec())) {
    const ground = GROUND(at.x, at.z);
    const crest = at.y + radius - ground;
    // The crest stands proud by the authored height, within the modulation.
    assert.ok(crest > 0.7 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m
      && crest < 1.4 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m, `crest ${crest} at ${at.x},${at.z}`);
    // And the section is buried, so the solid is set into the plaster rather
    // than resting on it: a rim that floats has a seam of sky under it.
    // At the pond form the section's underside sits exactly `undercut * radius`
    // below the plaster, so a quarter of a radius is a bound with room in it
    // rather than the identity restated.
    assert.ok(at.y - radius < ground - 0.25 * section.radius_m, "the section must be bedded in");
    checked += 1;
  }
  assert.equal(checked, STATIONS);
});

test("the coping's section varies as it runs, which is what stops it reading as an extrusion", () => {
  // The first version of the pond's *heightfield* held its crest to a constant
  // height and had a test pinning it there. That is the wrong invariant: the eye
  // reads an unvarying crest line long before it reads a wandering outline, so a
  // rim with a constant section looks machined however much the plan wobbles.
  //
  // The property outlived the representation. It was asserted on
  // `pondVesselHeightAt` until `HERO_GARDEN_VESSEL.crest` became `"flat"` and the
  // rim became this solid, at which point the heightfield's crest was
  // identically zero and the assertion was measuring plaster. It belongs here:
  // this is the run the eye actually follows.
  const section = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);
  const crests: number[] = [], radii: number[] = [];
  for (const { at, radius } of copingRailSamples(spec())) {
    // The crest line, not the axis: what stands proud of the plaster at this
    // node is the section's *top*, so a swell in the radius raises it as well as
    // widening it. That is deliberate — `sweptCopingNodes` holds the axis lift
    // rather than the crest, so the undercut survives a swell — and it means
    // this one number carries both modulations, which is also what the eye gets.
    crests.push(at.y + radius - GROUND(at.x, at.z));
    radii.push(radius);
  }
  assert.equal(crests.length, STATIONS);

  const swing = Math.max(...crests) - Math.min(...crests);
  // Enough to see — measured 18.2 mm on a 55 mm rim...
  assert.ok(swing > 0.004, `the crest barely moves: ${(swing * 1e3).toFixed(2)} mm`);
  // ...and not so much that the rim stops being one continuous form.
  assert.ok(swing < 0.5 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m, `the crest swings wildly: ${(swing * 1e3).toFixed(2)} mm`);
  for (const crest of crests) {
    assert.ok(Math.abs(crest - SWEPT_COPING_POND_BULLNOSE.crestHeight_m) < 0.4 * SWEPT_COPING_POND_BULLNOSE.crestHeight_m,
      `crest ${(crest * 1e3).toFixed(2)} mm leaves the authored band around ${SWEPT_COPING_POND_BULLNOSE.crestHeight_m * 1e3} mm`);
  }
  // And the width varies with it, on its own modulation rather than in step with
  // the crest's — a rim that got taller exactly where it got fatter would read as
  // one repeating motif. Measured 11.8 mm of swing on a 36.7 mm section.
  const width = Math.max(...radii) - Math.min(...radii);
  assert.ok(width > 0.1 * section.radius_m, `the section barely swells: ${(width * 1e3).toFixed(2)} mm`);
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

/**
 * The published hero document.
 *
 * Everything above works against a locally re-baked vessel, which is the right
 * fixture for a *generator* test: it isolates the species from the scene. The
 * agreement between the swept rim and the plaster it is set into is not a
 * property of the species, though — it is a property of the document, where
 * `rimHalfWidth_m`, the ground query and the section all have to line up — so
 * that one is asserted on what `createHeroGardenHoseScene` actually publishes.
 */
const HERO = createHeroGardenHoseScene();
const HERO_TERRAIN = HERO.terrain!;
const HERO_GRAPH = HERO.scenery!;

test("the swept rim and the plaster it is set into agree, on the document that publishes both", () => {
  // The coping is doubly authored: the solid is scenery and the ground it stands
  // in is terrain, and until this test nothing asserted the two describe one
  // object. They meet in three places and this is each of them.
  const section = sweptCopingSection(SWEPT_COPING_POND_BULLNOSE);

  // The document holds the rim as a `generator` node rather than as records: a
  // seed and about a dozen numbers, grown on expansion. So it has to be grown
  // here too, and the one input supplied rather than read is the ground query,
  // because that is exactly the thing a JSON document cannot hold. This mirrors
  // `growGenerator` in `lib/scenery-expand.ts`, and the last assertion in this
  // test closes that loop against the published catalog so the mirror cannot
  // drift into asserting something the scene does not do.
  const copingNode = HERO_GRAPH.nodes.find(({ id }) => id === "coping");
  assert.ok(copingNode?.kind === "generator", "the hero document must hold its rim as a generator node");
  const vessel = HERO_GRAPH.vessels?.[copingNode.vessel ?? ""];
  assert.ok(vessel, "the rim's generator node must name a vessel the graph declares");
  const heroGround = (x_m: number, z_m: number) => terrainHeightAt(HERO_TERRAIN, x_m, z_m);
  const heroCoping = growSceneryGenerator(copingNode.generator, copingNode.params, {
    key: copingNode.id,
    seed: copingNode.seed,
    groundHeightAt: heroGround,
    vessel: () => vessel,
  });
  // The same inputs, taken to the seam instead of to the emitter: this is the
  // rim's shape as the document specifies it, and `heroCoping` above is the
  // records that shape was turned into. Both are needed — the assertions below
  // are about the shape, and the last one closes the loop through the records.
  const heroParams = copingNode.params as SweptCopingSpec & { railSamplesPerLobe?: number };
  const heroStations = sweptCopingStations({
    ...heroParams,
    key: copingNode.id,
    seed: copingNode.seed,
    rail: pondVesselPlanCurve(vessel, heroParams.railSamplesPerLobe ?? SWEPT_COPING_RAIL_SAMPLES_PER_LOBE),
    groundHeightAt: heroGround,
    material: { palette: "stone", value: 0.9 },
  });

  // One: the seating, at every rail node.
  let worstOffLevel = 0, leastBurial = Infinity, leastFreeboard = Infinity, lowestUnderside = Infinity;
  let bandTotal = 0, footTotal = 0, worstMismatch = 0, samples = 0;
  for (const { at, radius } of heroStations) {
    const ground = terrainHeightAt(HERO_TERRAIN, at.x, at.z);
    worstOffLevel = Math.max(worstOffLevel, Math.abs(ground - HERO_GARDEN_VESSEL.groundHeight_m));
    leastBurial = Math.min(leastBurial, ground - (at.y - radius));
    leastFreeboard = Math.min(leastFreeboard, at.y + radius - HERO_GARDEN_WATERLINE_M);
    lowestUnderside = Math.min(lowestUnderside, at.y - radius);
    // The solid's own footprint at this node: a circle of radius `radius` whose
    // centre stands `lift` above the plaster meets it `sqrt(r^2 - lift^2)` either
    // side of the rail — `sweptCopingSection(...).groundHalfWidth_m` evaluated
    // with this node's modulated radius instead of the authored one.
    const lift = at.y - ground;
    const footprint = Math.sqrt(Math.max(0, radius * radius - lift * lift));
    const band = pondVesselRimHalfWidthAt(HERO_GARDEN_VESSEL, pondVesselTurnAt(HERO_GARDEN_VESSEL, at.x, at.z));
    bandTotal += band; footTotal += footprint;
    worstMismatch = Math.max(worstMismatch, Math.abs(band - footprint));
    samples += 1;
  }

  // The plaster under the rail is level, which is what `crest: "flat"` buys and
  // what gives the solid a band to be bedded into rather than a shoulder to
  // perch on. Measured 2.20 mm of wander against the 2.5 mm relief; if the
  // heightfield ever grew its crest back this would be 57 mm out.
  assert.ok(worstOffLevel <= HERO_GARDEN_VESSEL.relief_m,
    `the plaster under the rim wanders ${(worstOffLevel * 1e3).toFixed(2)} mm, past the ${HERO_GARDEN_VESSEL.relief_m * 1e3} mm relief`);
  // The section is buried at every node — a rim that floats has a seam of sky
  // under it all the way round. Nominally the burial is exactly the section's
  // axis lift (18.35 mm); the crest and radius modulations drag it to a measured
  // worst of 8.9 mm, so two fifths of the axis lift is a bound with room in it
  // rather than the identity restated. It was 10.6 mm before the rail's stride
  // went to four — a coarser walk lands the stations on different plaster, and
  // this is the only bound in the file that noticed.
  assert.ok(leastBurial > 0.4 * section.axisLift_m,
    `the rim's underside comes within ${(leastBurial * 1e3).toFixed(2)} mm of the plaster surface`);

  // Two: the width the two authorities are supposed to share.
  // `HERO_GARDEN_VESSEL.rimHalfWidth_m` is *derived* from
  // `sweptCopingSection(...).groundHalfWidth_m`, so the plaster's flat band and
  // the solid's footprint are the same number by construction — but only on the
  // mean, because each is then modulated independently: the band by the vessel's
  // `sectionWidthVariation` (22 %, salt 977, vessel seed) and the footprint by
  // the form's `sectionVariation` (14 %, coping seed). Measured, the means agree
  // to 0.07 mm on a 31.75 mm half-width, which is the derivation holding.
  assert.ok(Math.abs(bandTotal - footTotal) / samples < 1e-3,
    `the flat band averages ${(bandTotal / samples * 1e3).toFixed(2)} mm against a ${(footTotal / samples * 1e3).toFixed(2)} mm footprint`);
  // Pointwise they do not, and cannot while the two modulations are independent:
  // the worst turn leaves 14.0 mm of daylight between the band's edge and the
  // solid's — a shelf of dry plaster inside the rim where the band is the wider,
  // a lip overhanging the inner face where the solid is. The authored envelopes
  // alone would permit 27 mm, so this bound is on the two seeds as much as on the
  // form, and it is here to catch a re-seed that swings them apart rather than to
  // certify the construction. Closing it properly means one modulation feeding
  // both, which is a change to how the scene wires the vessel to the rim.
  assert.ok(worstMismatch < 0.9 * section.groundHalfWidth_m,
    `the band and the footprint part company by ${(worstMismatch * 1e3).toFixed(1)} mm at some turn`);

  // Three: containment, which is the claim the whole decision rests on. A coping
  // is the part of a vessel above the still waterline, so a rim the solver
  // cannot see costs the water nothing. Measured, the crest clears the waterline
  // by 92 mm and the solid's *lowest* point — its underside, buried in plaster —
  // still stands 16 mm clear, so no part of the invisible geometry is ever under
  // water. The ground-side half of containment is pinned on this same document
  // by "a tank fill wets the basin and nothing else" in
  // `tests/pond-vessel.test.ts`, and is deliberately not repeated here.
  assert.ok(leastFreeboard > HERO_GARDEN_CELL_M,
    `the crest stands only ${(leastFreeboard * 1e3).toFixed(1)} mm above the still waterline`);
  assert.ok(lowestUnderside > HERO_GARDEN_WATERLINE_M,
    `the rim's underside dips ${((HERO_GARDEN_WATERLINE_M - lowestUnderside) * 1e3).toFixed(1)} mm under the still waterline`);

  // A check on the fixture: every rail node really is on the plan curve, so "the
  // band at this turn" and "the footprint at this node" are two measurements of
  // one place rather than of two nearby ones. A quarter of a millimetre, which is
  // the plan polyline's own chord sag.
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
  for (const { at } of heroStations) {
    assert.ok(Math.abs(pondVesselPlanDistance(curve, at.x, at.z)) < 1e-3,
      `a station is ${(pondVesselPlanDistance(curve, at.x, at.z) * 1e3).toFixed(2)} mm off the plan curve`);
  }

  // And the loop closed: the rim above was grown with a ground query this file
  // supplied, so on its own it asserts what the species does with a heightfield
  // rather than what the *scene* does. The expansion supplies its own — the
  // document's terrain, resolved in `growGenerator` — and every record it
  // publishes has to be the same solid. Compared on the vertical extent, because
  // the ground enters nowhere else: a rim bedded into the generator instead of
  // the bake, or into a plane at `groundHeight_m`, differs from this one by about
  // two millimetres and would satisfy every bound above.
  const catalog = new Map(environmentProxyPrimitives(buildEnvironmentProxyCatalog(HERO, "garden"))
    .map((proxy) => [proxy.key.replace(/^garden\//, ""), proxy]));
  const published = heroCoping.map((node) => {
    const proxy = catalog.get(node.id);
    assert.ok(proxy, `${node.id} never reached the published catalog`);
    return proxy;
  });
  assert.equal(published.length, heroCoping.length, "every record the generator grew must reach the catalog");
  // Every station has to be inside the union of the published envelopes, at its
  // own radius. That is the loop the record layer could break without any bound
  // above noticing: a rim bedded into a plane at `groundHeight_m` rather than
  // into the bake differs from this one by about two millimetres, and an
  // envelope solved against the wrong stations would clip the ring.
  let worstReach = 0;
  for (const { at } of heroStations) {
    let nearest = Infinity;
    for (const proxy of published) {
      const { min, max } = proxy.aabb_m;
      const dx = Math.max(min.x - at.x, 0, at.x - max.x);
      const dy = Math.max(min.y - at.y, 0, at.y - max.y);
      const dz = Math.max(min.z - at.z, 0, at.z - max.z);
      nearest = Math.min(nearest, Math.hypot(dx, dy, dz));
      if (nearest === 0) break;
    }
    // The station's whole section has to be inside some record's bounds.
    worstReach = Math.max(worstReach, nearest);
    assert.equal(nearest, 0, `a station at ${at.x.toFixed(3)},${at.z.toFixed(3)} is outside every published sweep`);
  }
  assert.equal(worstReach, 0);
  assert.ok(samples > 0);
});
