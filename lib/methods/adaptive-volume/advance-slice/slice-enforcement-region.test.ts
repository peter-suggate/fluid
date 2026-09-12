import assert from "node:assert/strict";
import test from "node:test";
import { createDefaultSliceSceneSeed } from "./slice-scene-seed";
import {
  DEFAULT_SLICE_ENFORCEMENT_CELL_SIZE, clampSliceEnforcementCellSize,
  sliceEnforcementRegionAt, sliceEnforcementRegionCanvasBox,
  sliceEnforcementRegionFromCanvasDrag, sliceEnforcementRegions,
  sliceSceneRegions, withSliceEnforcementRegion,
} from "./slice-enforcement-region";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";

/** The fallback seed is [96, 40] finest cells at 0.05 m, origin at 0. */
const seed = () => createDefaultSliceSceneSeed();

/** Metres, to the nearest thousandth of a finest cell. */
function near(actual: number, expected: number, what: string): void {
  assert.ok(Math.abs(actual - expected) < 5e-5, `${what}: ${actual} != ${expected}`);
}

test("a drag snaps outward onto the ladder of its own minimum cell", () => {
  const region = sliceEnforcementRegionFromCanvasDrag(seed(), [9, 9], [22, 21],
    { minimumCellSize_cells: 4 });
  assert.equal(region.rule, "minimum-cell-size");
  assert.equal(region.minimumCellSize_cells, 4);
  // x: 9..22 grows out to 8..24. y is canvas-down over 40 cells, so the drag
  // spans source rows 19..31, which grows out to 16..32.
  near(region.min_m.x, 8 * 0.05, "left edge");
  near(region.max_m.x, 24 * 0.05, "right edge");
  near(region.min_m.y, 16 * 0.05, "lower edge");
  near(region.max_m.y, 32 * 0.05, "upper edge");
});

test("the box is never thinner than one cell it asks for, and stays on the lattice", () => {
  const region = sliceEnforcementRegionFromCanvasDrag(seed(), [3.2, 5], [3.4, 5.1],
    { minimumCellSize_cells: 8 });
  near(region.max_m.x - region.min_m.x, 8 * 0.05, "width");
  near(region.max_m.y - region.min_m.y, 8 * 0.05, "height");
  const atWall = sliceEnforcementRegionFromCanvasDrag(seed(), [94, 1], [99, -3],
    { minimumCellSize_cells: 8 });
  near(atWall.max_m.x, 96 * 0.05, "clamped to the far wall");
  near(atWall.max_m.y, 40 * 0.05, "clamped to the top row");
});

test("snapping is idempotent through the metres round trip", () => {
  const first = sliceEnforcementRegionFromCanvasDrag(seed(), [8, 8], [24, 24],
    { minimumCellSize_cells: 4 });
  const box = sliceEnforcementRegionCanvasBox(seed(), first);
  assert.ok(box);
  const second = sliceEnforcementRegionFromCanvasDrag(seed(),
    box.minFine, box.maxFine, { minimumCellSize_cells: 4, id: first.id });
  assert.deepEqual(second, first);
});

test("a drawn region covers this cut and only this cut", () => {
  const one = seed();
  const region = sliceEnforcementRegionFromCanvasDrag(one, [10, 10], [30, 30]);
  assert.equal(region.minimumCellSize_cells, DEFAULT_SLICE_ENFORCEMENT_CELL_SIZE);
  assert.ok(region.min_m.z < one.viewport.centerZ && one.viewport.centerZ < region.max_m.z,
    "the containment test the solver runs must accept it");
  near(region.max_m.z - region.min_m.z, one.viewport.sourceCellSize,
    "one source cell of depth, not a slab through a domain the picture cannot show");
  const next = withSliceEnforcementRegion(one, region.id, region);
  assert.deepEqual(sliceEnforcementRegions(next), [region]);
  assert.deepEqual(sliceEnforcementRegionAt(next, [20, 20]), region);
  assert.equal(sliceEnforcementRegionAt(next, [80, 20]), undefined);
});

test("a region is added, replaced and removed by id", () => {
  const one = seed();
  const first = sliceEnforcementRegionFromCanvasDrag(one, [4, 4], [12, 12]);
  const added = withSliceEnforcementRegion(one, first.id, first);
  const finer = { ...first, minimumCellSize_cells: 1 };
  const replaced = withSliceEnforcementRegion(added, first.id, finer);
  assert.equal(sliceSceneRegions(replaced).length, 1);
  assert.equal(sliceSceneRegions(replaced)[0]!.minimumCellSize_cells, 1);
  assert.deepEqual(sliceSceneRegions(withSliceEnforcementRegion(replaced, first.id, undefined)), []);
  assert.deepEqual(sliceSceneRegions(one), [], "the edit never touches the seed it came from");
});

test("the ladder stops where the slice's rungs do", () => {
  assert.equal(clampSliceEnforcementCellSize(32), 8);
  assert.equal(clampSliceEnforcementCellSize(3), 2);
  assert.equal(clampSliceEnforcementCellSize(0.5), 1);
});

test("the policy holds bricks the region contains at the size it names", () => {
  const one = seed();
  /* The whole lattice at one rung, so every brick is fully contained and the
   * floor is the only thing that can have decided the result. */
  const region = sliceEnforcementRegionFromCanvasDrag(one, [0, 0],
    [one.dimensions[0], one.dimensions[1]], { minimumCellSize_cells: 1,
      maximumCellSize_cells: 1 });
  const slice = createAdvanceSlice(withSliceEnforcementRegion(one, region.id, region));
  advanceSlice(slice, 8);
  const resolutions = new Set(slice.topology.accepted.bricks
    .filter(brick => brick.active !== false).map(brick => brick.resolution));
  assert.deepEqual([...resolutions], [8],
    "a brick of 8 finest cells held at 1-cell leaves is resolution 8");
});
