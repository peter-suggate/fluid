import assert from "node:assert/strict";
import test from "node:test";
import {
  refinementRegionSelectionId, regionCapacityRemaining, type RegionBox,
} from "../lib/features/refinement-region/definition";
import {
  moveRegionBox, regionCaption, regionSnapStep_cells, regionWithCeiling,
  regionWithFloor, resizeRegionBox, snapRegionBox,
} from "../lib/features/refinement-region/policy";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import { ADVANCE_BRICK_FINE } from "../lib/physics-wasm/advance-view";
import {
  labNextRegionId, labRegionAt, labRegionCanvasBox, labRegionFromCanvasBox,
  labRegionFromRecord, labRegionIsDrawn, labRegionRecord, labRegionSpace,
  labSolverCell, type LabRegionDocument,
} from "./lab-region-space";

/**
 * The lab as a `RegionSpace`: the frame, and nothing else.
 *
 * `advance-lab/slice-regions.ts` is gone, and with it eleven tests holding a
 * hand-written copy of the studio's region arithmetic — the outward snap, the
 * held opposite side, the translate-not-reshape move, the ceiling that follows
 * its floor, the selection-id prefix — to the studio's original. That was the
 * right test for two implementations and is the wrong test for one: the
 * arithmetic is `lib/features/refinement-region/policy.ts` now and is pinned
 * beside itself, and re-asserting it here would be asserting that a function
 * called twice returns the same thing.
 *
 * What is genuinely this adapter's, and is what this file holds:
 *
 *   - the **y-flip** between the solver's frame and the canvas, in both
 *     directions and round-trip;
 *   - the **field names** `AdvanceRefinementRegion` uses, in and out;
 *   - **which rungs, which brick, which capacity** this lattice declares;
 *   - the **topmost box** rule under a pointer, and what counts as drawn.
 *
 * The policy calls below are there to pin that the adapter hands the shared
 * functions the right frame — a resize stepping by the brick rather than by the
 * region's own floor, a caption reading in cells because this host has no metre
 * scale — not to re-test the functions themselves.
 */

const STORED: AdvanceRefinementRegion = {
  id: "advance-region-1",
  minimumFine: [16, 8],
  maximumFine: [48, 40],
  minimumCellWidth: 2,
};

const DOC: LabRegionDocument = { regions: [STORED], nx: 128, ny: 64 };

test("the adapter declares this lattice's ladder, brick and tail", () => {
  assert.equal(labRegionSpace.axes, 2);
  assert.deepEqual(labRegionSpace.cellSizes, [1, 2, 4, 8]);
  assert.equal(labRegionSpace.brick_cells, ADVANCE_BRICK_FINE);
  assert.equal(labRegionSpace.defaultCellSize_cells, 2,
    "the rung this lab shipped, not the studio's whole brick");
  assert.deepEqual(labRegionSpace.lattice(DOC), { dimensions: [128, 64] });
  assert.equal(labRegionSpace.cellEdge_mm, undefined,
    "no metre scale, so the shared rows omit the millimetre clause");
  assert.equal(regionCapacityRemaining(labRegionSpace, DOC), labRegionSpace.capacity - 1);
});

test("a stored box and the shared record are the same box, both ways", () => {
  const record = labRegionRecord(STORED);
  assert.deepEqual(record.min_cells, [16, 8]);
  assert.deepEqual(record.max_cells, [48, 40]);
  assert.equal(record.minimumCellSize_cells, 2);
  assert.equal(record.maximumCellSize_cells, undefined, "AUTO stays AUTO, not a fabricated ceiling");
  assert.deepEqual(labRegionFromRecord(record), STORED);

  const held = labRegionRecord({ ...STORED, maximumCellWidth: 2 });
  assert.equal(held.maximumCellSize_cells, 2);
  assert.deepEqual(labRegionFromRecord(held), { ...STORED, maximumCellWidth: 2 });
});

test("a box is stored with y up and painted with y down", () => {
  const record = labRegionRecord(STORED);
  const box = labRegionCanvasBox(record, DOC.ny);
  // The floor of the solver's box is the *bottom* of the picture.
  assert.deepEqual(box.min, [16, DOC.ny - 40]);
  assert.deepEqual(box.max, [48, DOC.ny - 8]);
  // And back again, unchanged: the flip is its own inverse.
  assert.deepEqual(labRegionFromCanvasBox(record, box, DOC.ny), record);
  // One point, the same reflection, so the snap can happen in the solver's frame.
  assert.deepEqual(labSolverCell([16, DOC.ny - 40], DOC.ny), [16, 40]);
});

test("the topmost box under the pointer is the one meant", () => {
  const under: AdvanceRefinementRegion = { ...STORED, id: "advance-region-2",
    minimumFine: [20, 12], maximumFine: [44, 36] };
  const records = labRegionSpace.list({ ...DOC, regions: [STORED, under] });
  // Both contain the point; the later one is drawn over the earlier one.
  assert.equal(labRegionAt(records, DOC.ny, [30, DOC.ny - 20])?.id, "advance-region-2");
  assert.equal(labRegionAt(records, DOC.ny, [17, DOC.ny - 9])?.id, "advance-region-1");
  assert.equal(labRegionAt(records, DOC.ny, [0, 0]), undefined);
  assert.equal(labRegionAt(records, DOC.ny, null), undefined);
});

test("a released click draws no box", () => {
  assert.equal(labRegionIsDrawn(labRegionRecord(STORED)), true);
  assert.equal(labRegionIsDrawn({ ...labRegionRecord(STORED), max_cells: [16, 8] }), false);
});

test("writing replaces, adds and drops, leaving the lattice alone", () => {
  const record = labRegionRecord(STORED);
  const moved = { ...record, min_cells: [0, 0], max_cells: [16, 16] };
  const replaced = labRegionSpace.write(DOC, record.id, moved);
  assert.deepEqual(replaced.regions.map(region => region.id), ["advance-region-1"]);
  assert.deepEqual(replaced.regions[0]?.minimumFine, [0, 0]);
  assert.equal(replaced.nx, DOC.nx);

  const added = labRegionSpace.write(DOC, "advance-region-2",
    { ...record, id: "advance-region-2" });
  assert.deepEqual(added.regions.map(region => region.id),
    ["advance-region-1", "advance-region-2"]);
  assert.deepEqual(labRegionSpace.write(DOC, record.id, undefined).regions, []);
});

test("the next id names its position in the list the reader is looking at", () => {
  assert.equal(labNextRegionId({ ...DOC, regions: [] }), "advance-region-1");
  assert.equal(labNextRegionId(DOC), "advance-region-2");
  // A counter in a ref survived a scene change that emptied the list; reading
  // the highest suffix back cannot.
  assert.equal(labNextRegionId({ ...DOC, regions: [
    { ...STORED, id: "advance-region-7" }] }), "advance-region-8");
});

test("a resize steps by the brick, not by the region's own floor", () => {
  const record = labRegionRecord(STORED);
  const step = regionSnapStep_cells(record, labRegionSpace.brick_cells);
  assert.equal(step, ADVANCE_BRICK_FINE,
    "a floor of 2 still snaps to 8: the solver binds a region brick by brick");
  const box: RegionBox = { min: record.min_cells, max: record.max_cells };
  const wider = resizeRegionBox(box, { axis: 0, end: "max" }, 61, step,
    labRegionSpace.lattice(DOC));
  assert.deepEqual(wider.max, [64, 40], "the edge taken hold of moves, onto the brick");
  assert.deepEqual(wider.min, [16, 8], "and the opposite one is held");

  const nudged = moveRegionBox(box, [3, 0], step, labRegionSpace.lattice(DOC));
  assert.deepEqual([nudged.min[0], nudged.max[0]], [16, 48],
    "a move translates by whole steps and never reshapes");
});

test("a drawn box snaps outward in the solver's frame and stops at the wall", () => {
  const lattice = labRegionSpace.lattice(DOC);
  const snapped = snapRegionBox([10, 10], [70, 50], ADVANCE_BRICK_FINE, lattice);
  assert.deepEqual(snapped.min, [8, 8]);
  assert.deepEqual(snapped.max, [72, 56]);
  const atWall = snapRegionBox([120, 50], [127, 63], ADVANCE_BRICK_FINE, lattice);
  assert.deepEqual(atWall.max, [128, 64], "the lattice wins over the step at the wall");
});

test("a ceiling equal to its floor follows the floor, and AUTO removes it", () => {
  const held = { ...labRegionRecord(STORED), maximumCellSize_cells: 2 };
  const lattice = labRegionSpace.lattice(DOC);
  const raised = regionWithFloor(held, 8, lattice, labRegionSpace.brick_cells);
  assert.equal(raised.maximumCellSize_cells, 8, "the held ceiling follows the floor up");
  const lowered = regionWithFloor(raised, 1, lattice, labRegionSpace.brick_cells);
  assert.equal(lowered.maximumCellSize_cells, 1, "and down, which the studio alone did not");
  assert.equal(regionWithCeiling(held, undefined).maximumCellSize_cells, undefined);
});

test("the tag reads in cells, because this host has no metres to quote", () => {
  assert.equal(regionCaption(labRegionRecord(STORED)), "≥ 2 cells");
  assert.equal(regionCaption({ ...labRegionRecord(STORED), maximumCellSize_cells: 2 }),
    "held at 2");
  assert.equal(regionCaption({ ...labRegionRecord(STORED), minimumCellSize_cells: 1 }),
    "≥ 1 cell");
});

test("a region selected here is the same kind of selection as one selected in the studio", () => {
  assert.equal(refinementRegionSelectionId("advance-region-1"),
    "refinement-region-advance-region-1");
});
