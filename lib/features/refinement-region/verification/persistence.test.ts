import assert from "node:assert/strict";
import test from "node:test";
import {
  REGIONS_QUERY_KEY, regionsFromQuery, regionsToQuery,
} from "../persistence";
import {
  BRICK_FINE_CELLS,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  OCTREE_REFINEMENT_REGION_CELL_SIZES,
  type RefinementRegionRecord,
  type RegionSpace,
} from "../definition";

/**
 * The `regions=` wire form, in the one frame both hosts share.
 *
 * Nothing here mentions metres, a scene document or a canvas — the encoding is
 * percentages of a lattice and counts of finest cells, in N axes — which is the
 * claim under test as much as any single assertion: the studio spelled this out
 * against `Vec3` and the 2-D lab had no address bar at all, and the reason a
 * link can now be written by either is that there is one codec rather than two.
 */

/** A host with `axes` axes over a fixed lattice, and the shipped ladders. */
function space(axes: 2 | 3, dimensions: readonly number[],
  regions: readonly RefinementRegionRecord[] = []):
  Pick<RegionSpace<null, unknown>,
    "axes" | "lattice" | "list" | "capacity" | "brick_cells"
    | "cellSizes" | "defaultCellSize_cells"> {
  return {
    axes,
    lattice: () => ({ dimensions }),
    list: () => regions,
    capacity: OCTREE_REFINEMENT_REGION_CAPACITY,
    brick_cells: BRICK_FINE_CELLS,
    cellSizes: OCTREE_REFINEMENT_REGION_CELL_SIZES,
    defaultCellSize_cells: 8,
  };
}

const box = (min: readonly number[], max: readonly number[],
  floor = 8, ceiling?: number): RefinementRegionRecord => ({
  id: "region-1",
  rule: "minimum-cell-size",
  minimumCellSize_cells: floor,
  ...(ceiling === undefined ? {} : { maximumCellSize_cells: ceiling }),
  min_cells: [...min],
  max_cells: [...max],
});

test("the key is the feature's, so both hosts spell it the same way", () => {
  assert.equal(REGIONS_QUERY_KEY, "regions");
});

test("a whole-domain box is 0..100 on every axis, in both worlds", () => {
  assert.equal(
    regionsToQuery(space(3, [128, 128, 128], [box([0, 0, 0], [128, 128, 128], 2)]), null),
    "0_0_0_100_100_100_2");
  assert.equal(
    regionsToQuery(space(2, [96, 64], [box([0, 0], [96, 64], 2)]), null),
    "0_0_100_100_2");
});

test("the shipped three-axis link still parses to the whole domain", () => {
  // The literal `tests/compare-model.test.ts` has carried since before the
  // package existed. A link is forever: this is the one assertion that says so.
  const parsed = regionsFromQuery(space(3, [64, 48, 64]), null, "0_0_0_100_100_100_2");
  assert.equal(parsed.length, 1);
  assert.deepEqual(parsed[0]!.min_cells, [0, 0, 0]);
  assert.deepEqual(parsed[0]!.max_cells, [64, 48, 64]);
  assert.equal(parsed[0]!.minimumCellSize_cells, 2);
  assert.equal(parsed[0]!.maximumCellSize_cells, undefined);
  assert.equal(parsed[0]!.id, "region-1");
});

test("a record round-trips through the wire unchanged", () => {
  // Dyadic lattices, because the wire carries four decimal places of a
  // percentage: a side at cell 8 of 96 is 8.3333%, which reads back a third of a
  // thousandth of a cell short and snaps *down* a whole step. That granularity
  // is the format's and predates the package — a link is a hundredth of a
  // percent of the domain, not a cell address — and the snap is what keeps it
  // from mattering: every side a reader can place is a step multiple.
  for (const [axes, dimensions] of [[3, [128, 128, 128]], [2, [128, 64]]] as const) {
    const record = axes === 3
      ? box([16, 8, 24], [48, 32, 64], 8, 16)
      : box([16, 8], [48, 32], 8, 16);
    const wire = regionsToQuery(space(axes, dimensions, [record]), null);
    const back = regionsFromQuery(space(axes, dimensions), null, wire);
    assert.equal(back.length, 1);
    assert.deepEqual(back[0]!.min_cells, record.min_cells);
    assert.deepEqual(back[0]!.max_cells, record.max_cells);
    assert.equal(back[0]!.minimumCellSize_cells, 8);
    assert.equal(back[0]!.maximumCellSize_cells, 16);
    assert.equal(regionsToQuery(space(axes, dimensions, back), null), wire);
  }
});

test("the same record is the same string whatever the host measuring it", () => {
  // Two adapters, one lattice shape on the axes they share: the 2-D lab and the
  // 3-D studio must write the same percentages for the same box, or a link
  // would mean one thing on one page and another on the other.
  const shared = regionsToQuery(space(2, [128, 96], [box([32, 24], [64, 48], 8)]), null);
  const spatial = regionsToQuery(
    space(3, [128, 96, 64], [box([32, 24, 0], [64, 48, 64], 8)]), null);
  assert.equal(shared, "25_25_50_50_8");
  assert.equal(spatial, "25_25_0_50_50_100_8");
});

test("a box off the lattice is snapped outward onto its own step", () => {
  // 3 to 61 finest cells at a floor of 8: the step is the brick, so the box
  // grows to 0..64 rather than keeping a shell of partly-covered cells.
  const parsed = regionsFromQuery(space(2, [128, 128]), null, "2.34375_2.34375_47.65625_47.65625_8");
  assert.deepEqual(parsed[0]!.min_cells, [0, 0]);
  assert.deepEqual(parsed[0]!.max_cells, [64, 64]);
});

test("a re-parse of a written value changes nothing", () => {
  const lattice = [128, 128, 128];
  const once = regionsFromQuery(space(3, lattice), null, "12_8_16_52_40_64_8");
  const wire = regionsToQuery(space(3, lattice, once), null);
  const twice = regionsFromQuery(space(3, lattice), null, wire);
  assert.deepEqual(twice, once);
  assert.equal(regionsToQuery(space(3, lattice, twice), null), wire);
});

test("both optional tails are accepted, and the default rule is never written", () => {
  const three = space(3, [64, 64, 64]);
  // Seven fields: no ceiling, no rule.
  assert.equal(regionsFromQuery(three, null, "0_0_0_50_50_50_8")[0]!
    .maximumCellSize_cells, undefined);
  // Eight fields, a number: the ceiling.
  assert.equal(regionsFromQuery(three, null, "0_0_0_50_50_50_8_16")[0]!
    .maximumCellSize_cells, 16);
  // Eight fields, the default rule spelled out: the old form, still read.
  const legacy = regionsFromQuery(three, null, "0_0_0_50_50_50_8_minimum-cell-size");
  assert.equal(legacy.length, 1);
  assert.equal(legacy[0]!.maximumCellSize_cells, undefined);
  // Nine fields: ceiling then rule.
  assert.equal(regionsFromQuery(three, null, "0_0_0_50_50_50_8_16_minimum-cell-size")[0]!
    .maximumCellSize_cells, 16);
  assert.equal(regionsToQuery(space(3, [64, 64, 64],
    regionsFromQuery(three, null, "0_0_0_50_50_50_8_16_minimum-cell-size")), null),
  "0_0_0_50_50_50_8_16");
});

test("a ceiling below the floor is raised to it rather than believed", () => {
  const parsed = regionsFromQuery(space(2, [64, 64]), null, "0_0_50_50_16_4");
  assert.equal(parsed[0]!.minimumCellSize_cells, 16);
  assert.equal(parsed[0]!.maximumCellSize_cells, 16);
});

test("one bad record costs the reader only that record", () => {
  const parsed = regionsFromQuery(space(2, [64, 64]), null,
    "0_0_25_25_8*not_a_box*50_50_100_100_8");
  assert.equal(parsed.length, 2);
  assert.deepEqual(parsed.map((region) => region.id), ["region-1", "region-2"]);
});

test("a rule nobody ships is dropped, not coerced", () => {
  assert.deepEqual(
    regionsFromQuery(space(3, [64, 64, 64]), null, "0_0_0_50_50_50_8_something-else"), []);
});

test("percentages outside the domain are clamped to it", () => {
  const parsed = regionsFromQuery(space(2, [64, 64]), null, "-40_-40_180_180_8");
  assert.deepEqual(parsed[0]!.min_cells, [0, 0]);
  assert.deepEqual(parsed[0]!.max_cells, [64, 64]);
});

test("a link cannot exceed the host's capacity", () => {
  const one = "0_0_50_50_8";
  const many = Array.from({ length: OCTREE_REFINEMENT_REGION_CAPACITY + 4 }, () => one)
    .join("*");
  assert.equal(regionsFromQuery(space(2, [64, 64]), null, many).length,
    OCTREE_REFINEMENT_REGION_CAPACITY);
});

test("a record that snapped to nothing is dropped", () => {
  // A lattice with no room for one step on the second axis: the snap cannot
  // produce a box, so there is nothing to draw and nothing to enforce.
  assert.deepEqual(regionsFromQuery(space(2, [64, 0]), null, "0_0_50_50_8"), []);
});
