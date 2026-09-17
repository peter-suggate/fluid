import assert from "node:assert/strict";
import test from "node:test";
import {
  moveRegionBox,
  regionCaption,
  regionChoices,
  regionSnapStep_cells,
  regionWithCeiling,
  regionWithFloor,
  resizeRegionBox,
  snapRegionBox,
} from "../policy";
import {
  BRICK_FINE_CELLS,
  type RefinementRegionRecord,
  type RegionLattice,
  type RegionSpace,
} from "../definition";

/**
 * The region policy, in the one frame both hosts share.
 *
 * Nothing here mentions metres, a scene document or a canvas, which is the
 * claim being tested as much as any single assertion: the rules the 3-D editor
 * and the 2-D lab each used to own are arithmetic on finest cells, and if one
 * of them still needed its host's units this file could not have been written.
 */

const lattice3: RegionLattice = { dimensions: [64, 48, 32] };
const lattice2: RegionLattice = { dimensions: [48, 24] };

function record(patch: Partial<RefinementRegionRecord> = {}): RefinementRegionRecord {
  return {
    id: "region-1",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
    min_cells: [8, 0, 8],
    max_cells: [24, 16, 24],
    ...patch,
  };
}

// (a) The snap default — the whole of the change Peter asked for.
test("the snap step is the brick until the floor is coarser than one", () => {
  const ladder = [1, 2, 4, 8, 16, 32];
  const steps = ladder.map((cells) =>
    regionSnapStep_cells({ minimumCellSize_cells: cells }, BRICK_FINE_CELLS));
  assert.deepEqual(steps, [8, 8, 8, 8, 16, 32]);
});

test("a host that binds per cell rather than per brick still gets its own step", () => {
  // `brick_cells` is a `RegionSpace` field, not a constant, so the finer step is
  // one adapter away rather than a UI toggle away.
  assert.equal(regionSnapStep_cells({ minimumCellSize_cells: 1 }, 1), 1);
  assert.equal(regionSnapStep_cells({ minimumCellSize_cells: 4 }, 1), 4);
});

test("an off-ladder floor rounds down to a power of two before the step is taken", () => {
  assert.equal(regionSnapStep_cells({ minimumCellSize_cells: 24 }, BRICK_FINE_CELLS), 16);
  assert.equal(regionSnapStep_cells({ minimumCellSize_cells: 0 }, BRICK_FINE_CELLS), 8);
  assert.equal(regionSnapStep_cells({ minimumCellSize_cells: Number.NaN }, BRICK_FINE_CELLS), 8);
});

// (b) A drawn box snaps outward, and stays put once it has.
test("a drawn box grows to cover every cell the drag touched", () => {
  const snapped = snapRegionBox([9.2, 0.4, 3.9], [17.1, 9.5, 12.2], 8, lattice3);
  assert.deepEqual(snapped.min, [8, 0, 0]);
  assert.deepEqual(snapped.max, [24, 16, 16]);
});

test("snapping is idempotent, so a round trip through a document cannot grow a box", () => {
  let box = snapRegionBox([9.2, 0.4, 3.9], [17.1, 9.5, 12.2], 8, lattice3);
  for (let pass = 0; pass < 5; pass += 1) {
    // The float noise a metres round trip leaves behind: an aligned edge comes
    // back as 8.000000001, and a bare `ceil` would add a whole brick each pass.
    const jittered = {
      min: box.min.map((value) => value + 1e-9),
      max: box.max.map((value) => value + 1e-9),
    };
    const next = snapRegionBox(jittered.min, jittered.max, 8, lattice3);
    assert.deepEqual(next.min, box.min);
    assert.deepEqual(next.max, box.max);
    box = next;
  }
});

test("the lattice wins over the step at the far wall", () => {
  // 48 is not a multiple of 32, so there is no aligned line on the wall; the box
  // stops at the wall rather than at the last line inside it.
  const snapped = snapRegionBox([40, 4], [47, 20], 32, { dimensions: [48, 24] });
  assert.deepEqual(snapped.max, [48, 24]);
  assert.deepEqual(snapped.min, [16, 0]);
});

test("a click with no area still yields one step of box", () => {
  const snapped = snapRegionBox([12, 12, 12], [12, 12, 12], 8, lattice3);
  for (let axis = 0; axis < 3; axis += 1) {
    assert.equal(snapped.max[axis]! - snapped.min[axis]!, 8, `axis ${axis}`);
  }
});

// (c) A resize never inverts and never leaves the lattice.
test("a resized side snaps, stays on the lattice and never crosses the held one", () => {
  const box = { min: [8, 0, 8], max: [24, 16, 24] };
  assert.deepEqual(resizeRegionBox(box, { axis: 0, end: "max" }, 41, 8, lattice3).max,
    [40, 16, 24]);
  // Dragged past the far wall.
  assert.deepEqual(resizeRegionBox(box, { axis: 1, end: "max" }, 900, 8, lattice3).max,
    [24, 48, 24]);
  // Dragged past the held side: one step of thickness is the floor.
  const inverted = resizeRegionBox(box, { axis: 0, end: "min" }, 400, 8, lattice3);
  assert.deepEqual(inverted.min, [16, 0, 8]);
  assert.ok(inverted.max[0]! - inverted.min[0]! >= 8);
  const other = resizeRegionBox(box, { axis: 2, end: "max" }, -50, 8, lattice3);
  assert.equal(other.max[2]! - other.min[2]!, 8);
});

test("a moved box keeps its size and stops against the wall", () => {
  const box = { min: [8, 0], max: [24, 8] };
  assert.deepEqual(moveRegionBox(box, [9, 3], 8, lattice2),
    { min: [16, 0], max: [32, 8] });
  const pushed = moveRegionBox(box, [500, 500], 8, lattice2);
  assert.deepEqual(pushed, { min: [32, 16], max: [48, 24] });
});

// The two ladders, and how they follow each other.
test("a ceiling held at one tier follows its floor in both directions", () => {
  const held = record({ minimumCellSize_cells: 8, maximumCellSize_cells: 8 });
  assert.equal(regionWithFloor(held, 16).maximumCellSize_cells, 16);
  assert.equal(regionWithFloor(held, 4).maximumCellSize_cells, 4);

  const wide = record({ minimumCellSize_cells: 4, maximumCellSize_cells: 16 });
  // Kept where it still clears the floor…
  assert.equal(regionWithFloor(wide, 8).maximumCellSize_cells, 16);
  // …and only ever lifted to stay above it.
  assert.equal(regionWithFloor(wide, 32).maximumCellSize_cells, 32);

  assert.equal(regionWithFloor(record(), 16).maximumCellSize_cells, undefined);
});

test("AUTO is the absence of a ceiling, not a ceiling holding nothing", () => {
  const capped = record({ maximumCellSize_cells: 16 });
  const auto = regionWithCeiling(capped, undefined);
  assert.equal("maximumCellSize_cells" in auto, false);
});

test("a ceiling under the floor moves the whole interval down", () => {
  const next = regionWithCeiling(record({ minimumCellSize_cells: 16 }), 4);
  assert.equal(next.minimumCellSize_cells, 4);
  assert.equal(next.maximumCellSize_cells, 4);
});

test("changing the floor re-snaps the box onto the step the new floor implies", () => {
  const before = record({ minimumCellSize_cells: 8, min_cells: [8, 0, 8], max_cells: [24, 16, 24] });
  const after = regionWithFloor(before, 32, lattice3, BRICK_FINE_CELLS);
  assert.deepEqual(after.min_cells, [0, 0, 0]);
  assert.deepEqual(after.max_cells, [32, 32, 32]);
  // A floor below the brick leaves the box exactly where the brick put it.
  assert.deepEqual(regionWithFloor(before, 1, lattice3, BRICK_FINE_CELLS).min_cells, before.min_cells);
});

test("a caption reads in millimetres where there is a scale and in cells where there is not", () => {
  const edge_mm = 5;
  assert.equal(regionCaption(record(), edge_mm),
    "No pressure cell smaller than 40 mm inside this box. "
    + "Grading still splits leaves on its boundary.");
  assert.equal(regionCaption(record({ maximumCellSize_cells: 8 }), edge_mm),
    "Fully contained pressure cells are held at 40 mm inside this box.");
  assert.equal(regionCaption(record({ maximumCellSize_cells: 32 }), edge_mm),
    "Fully contained pressure cells stay between 40 and 160 mm inside this box.");
  assert.equal(regionCaption(record({ maximumCellSize_cells: 8 })), "held at 8");
  assert.equal(regionCaption(record()), "≥ 8 cells");
  assert.equal(regionCaption(record({ minimumCellSize_cells: 1 })), "≥ 1 cell");
});

// (d) The same record through a 2-D adapter and a 3-D one.
//
// A pair of deliberately minimal spaces: what is being pinned is that the
// policy is indifferent to the axis count and to what a host calls a document,
// not anything about either real adapter.
function listSpace(
  axes: 2 | 3,
  lattice: RegionLattice,
  cellSizes: readonly number[],
): RegionSpace<readonly RefinementRegionRecord[], readonly RefinementRegionRecord[]> {
  return {
    axes,
    lattice: () => lattice,
    list: (doc) => doc,
    write: (doc, id, next) => {
      const replaced = doc.some((entry) => entry.id === id);
      if (!replaced) return next ? [...doc, next] : doc;
      return doc.flatMap((entry) => entry.id !== id ? [entry] : next ? [next] : []);
    },
    nextId: (doc) => `region-${doc.length + 1}`,
    capacity: 8,
    cellSizes,
    defaultCellSize_cells: cellSizes[cellSizes.length - 1] ?? BRICK_FINE_CELLS,
    brick_cells: BRICK_FINE_CELLS,
  };
}

test("two adapters of different dimension round-trip one record's edits alike", () => {
  const flat = record({ min_cells: [8, 0], max_cells: [24, 16] });
  const solid = record();
  const space2 = listSpace(2, lattice2, [1, 2, 4, 8]);
  const space3 = listSpace(3, lattice3, [1, 2, 4, 8, 16, 32]);

  const rows2 = regionChoices(space2, [flat], flat);
  const rows3 = regionChoices(space3, [solid], solid);
  // Same groups, same order, same tags — the row tree a host renders.
  assert.deepEqual(rows2.map((group) => [group.id, group.tag, group.value]),
    rows3.map((group) => [group.id, group.tag, group.value]));
  // The ladders differ because the hosts' ladders differ, and nothing else does.
  assert.deepEqual(rows2[1]!.options.map((option) => option.id), ["1", "2", "4", "8"]);
  assert.deepEqual(rows3[1]!.options.map((option) => option.id), ["1", "2", "4", "8", "16", "32"]);

  const apply2 = rows2[1]!.options.find((option) => option.id === "4")!.apply();
  const apply3 = rows3[1]!.options.find((option) => option.id === "4")!.apply();
  assert.equal(apply2.length, 1);
  assert.equal(apply3.length, 1);
  assert.equal(apply2[0]!.minimumCellSize_cells, 4);
  assert.equal(apply3[0]!.minimumCellSize_cells, 4);
  // Two axes out, three axes out: the record keeps the host's dimension.
  assert.equal(apply2[0]!.min_cells.length, 2);
  assert.equal(apply3[0]!.min_cells.length, 3);
  // A floor below the brick leaves the box on the brick lattice in both.
  assert.deepEqual(apply2[0]!.min_cells, [8, 0]);
  assert.deepEqual(apply3[0]!.min_cells, [8, 0, 8]);
});

test("a hint quotes a physical edge only where the host has one", () => {
  const withScale: RegionSpace<readonly RefinementRegionRecord[], readonly RefinementRegionRecord[]> = {
    ...listSpace(3, lattice3, [8]),
    cellEdge_mm: () => 5,
  };
  const withoutScale = listSpace(2, lattice2, [8]);
  const solid = record();
  assert.equal(regionChoices(withScale, [solid], solid)[1]!.options[0]!.hint,
    "8³ finest cells · 40 mm edge");
  assert.equal(regionChoices(withoutScale, [solid], solid)[1]!.options[0]!.hint,
    "8³ finest cells");
});
