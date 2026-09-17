import assert from "node:assert/strict";
import test from "node:test";
import { REFINEMENT_REGION_SELECTION_PREFIX } from "../lib/core/editor-refinement-region";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import {
  SLICE_REGION_SELECTION_PREFIX, SLICE_RESIZE_HANDLES, draftRegionBox, handleAnchor,
  movedRegionBox, regionAt, regionBox, regionBoxIsDrawn, regionCaption, regionWithBox,
  regionWithCeiling, regionWithFloor, resizedRegionBox, sliceRegionIdFromSelection,
  sliceRegionSelectionId,
  type SliceBox, type SliceLattice,
} from "./slice-regions";

/**
 * The enforcement box, as arithmetic.
 *
 * All of this is the 3-D editor's behaviour restated in two dimensions, so
 * every test here is really a claim that the lab's box behaves the way the
 * studio's does: outward snapping, a held opposite side, a move that translates
 * rather than reshapes, and a ceiling that follows its floor.
 */

const LATTICE: SliceLattice = { nx: 64, ny: 32 };

function region(patch: Partial<AdvanceRefinementRegion> = {}): AdvanceRefinementRegion {
  return {
    id: "advance-region-1",
    minimumFine: [8, 4],
    maximumFine: [24, 16],
    minimumCellWidth: 2,
    ...patch,
  };
}

test("the selection id mirrors the studio's, prefix for prefix", () => {
  /* Mirrored rather than imported at the source: `editor-refinement-region.ts`
   * drags the whole 3-D scene document behind it — entities, the model, the
   * region list — and none of that exists in the lab. Pinned here instead, so
   * the day the studio renames its prefix this fails rather than drifting. */
  assert.equal(SLICE_REGION_SELECTION_PREFIX, REFINEMENT_REGION_SELECTION_PREFIX);
  assert.equal(sliceRegionSelectionId("advance-region-2"), "refinement-region-advance-region-2");
  assert.equal(sliceRegionIdFromSelection("refinement-region-advance-region-2"),
    "advance-region-2");
  assert.equal(sliceRegionIdFromSelection("voxel-region"), undefined);
  assert.equal(sliceRegionIdFromSelection(undefined), undefined);
});

test("a box is drawn with y down and stored with y up", () => {
  // The one place the flip happens, and the reason it is one place.
  const box = regionBox(region(), LATTICE.ny);
  assert.deepEqual(box, { minFine: [8, 16], maxFine: [24, 28] });
  const back = regionWithBox(region(), box, LATTICE.ny);
  assert.deepEqual(back.minimumFine, [8, 4]);
  assert.deepEqual(back.maximumFine, [24, 16]);
});

test("the topmost box under the pointer is the one meant", () => {
  const under = region({ id: "under", minimumFine: [0, 0], maximumFine: [32, 32] });
  const over = region({ id: "over", minimumFine: [8, 8], maximumFine: [16, 16] });
  // Drawn later means drawn on top, and what a reader sees is what they mean.
  assert.equal(regionAt([under, over], LATTICE.ny, [12, 20])?.id, "over");
  assert.equal(regionAt([under, over], LATTICE.ny, [2, 20])?.id, "under");
  assert.equal(regionAt([under, over], LATTICE.ny, [40, 2])?.id, undefined);
  assert.equal(regionAt([under, over], LATTICE.ny, null), undefined);
});

test("a draft snaps outward and stops at the wall", () => {
  const box = draftRegionBox([3.2, 5.9], [10.1, 14.4], 4, LATTICE);
  // Outward, not nearest: a box smaller than the rectangle let go of is not the
  // instruction the reader gave.
  assert.deepEqual(box, { minFine: [0, 4], maxFine: [12, 16] });
  const past = draftRegionBox([-9, -9], [200, 200], 4, LATTICE);
  assert.deepEqual(past, { minFine: [0, 0], maxFine: [64, 32] });
});

test("a released click draws no box", () => {
  assert.equal(regionBoxIsDrawn(draftRegionBox([8, 8], [8, 8], 4, LATTICE)), false);
  assert.equal(regionBoxIsDrawn({ minFine: [0, 0], maxFine: [4, 4] }), true);
});

test("a resize moves the edge taken hold of and holds the opposite one", () => {
  const box: SliceBox = { minFine: [8, 8], maxFine: [24, 24] };
  const east = resizedRegionBox(box, "e", [31.2, 99], 4, LATTICE);
  assert.deepEqual(east, { minFine: [8, 8], maxFine: [32, 24] });
  const corner = resizedRegionBox(box, "nw", [3.1, 1.9], 4, LATTICE);
  assert.deepEqual(corner, { minFine: [4, 0], maxFine: [24, 24] });
  // One step of thickness is the floor: a region thinner than the cell it asks
  // for contains none of them.
  const collapsed = resizedRegionBox(box, "w", [99, 0], 4, LATTICE);
  assert.equal(collapsed.maxFine[0] - collapsed.minFine[0], 4);
});

test("a move translates and stops against the wall at its own size", () => {
  const box: SliceBox = { minFine: [8, 8], maxFine: [24, 24] };
  assert.deepEqual(movedRegionBox(box, 4, -4, 4, LATTICE),
    { minFine: [12, 4], maxFine: [28, 20] });
  const pushed = movedRegionBox(box, 500, 500, 4, LATTICE);
  assert.equal(pushed.maxFine[0] - pushed.minFine[0], 16, "a box squashed by a wall is a bug");
  assert.deepEqual(pushed, { minFine: [48, 16], maxFine: [64, 32] });
});

test("every handle has a corner and the set is the eight plus the body", () => {
  assert.deepEqual([...SLICE_RESIZE_HANDLES].sort(),
    ["e", "n", "ne", "nw", "s", "se", "sw", "w"]);
  assert.deepEqual(handleAnchor("nw"), [0, 0]);
  assert.deepEqual(handleAnchor("se"), [1, 1]);
  assert.deepEqual(handleAnchor("n"), [0.5, 0]);
  assert.deepEqual(handleAnchor("body"), [0.5, 0.5]);
});

test("a ceiling equal to its floor follows the floor", () => {
  // "Hold at one tier" is Max = Min, so the lab's old switch is a consequence
  // rather than a second piece of state.
  const held = regionWithFloor(region({ maximumCellWidth: 2 }), 4);
  assert.equal(held.maximumCellWidth, 4);
  // A wider authored ceiling is kept, and only lifted to stay above the floor.
  assert.equal(regionWithFloor(region({ maximumCellWidth: 8 }), 4).maximumCellWidth, 8);
  assert.equal(regionWithFloor(region({ maximumCellWidth: 2 }), 8).maximumCellWidth, 8);
  // No ceiling stays no ceiling: AUTO is the absence of one, not a large number.
  assert.equal("maximumCellWidth" in regionWithFloor(region(), 4), false);
});

test("AUTO removes the ceiling rather than blanking it", () => {
  const auto = regionWithCeiling(region({ maximumCellWidth: 4 }), undefined);
  assert.equal("maximumCellWidth" in auto, false);
  assert.equal(regionWithCeiling(region({ minimumCellWidth: 4 }), 2).maximumCellWidth, 4,
    "a ceiling below the floor is not a bound, it is a contradiction");
});

test("the tag says what the box enforces", () => {
  assert.equal(regionCaption(region({ minimumCellWidth: 2, maximumCellWidth: 2 })), "held at 2");
  assert.equal(regionCaption(region({ minimumCellWidth: 1 })), "≥ 1 cell");
  assert.equal(regionCaption(region({ minimumCellWidth: 4 })), "≥ 4 cells");
});
