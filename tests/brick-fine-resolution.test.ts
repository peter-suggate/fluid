import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_FINE_CELLS } from "../lib/core/sparse-brick-geometry";
import { BRICK_FINE_RESOLUTION, DEFAULT_BRICK_FINE_RESOLUTION } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { ADVANCE_BRICK_FINE } from "../lib/physics-wasm/advance-view";

/**
 * One brick width, three zones.
 *
 * The 3-D solver, the 2-D lab's view model and the editor's snap all have to
 * agree about how wide a brick is, and until now each held its own literal 8.
 * A region snapped to one of them and bound by another is a region whose floor
 * silently applies to cells it does not cover, so the equality is the invariant
 * rather than the value: if the ladder ever moves, it moves in `lib/core`, and
 * this test is what makes a second copy fail rather than drift.
 */
test("every zone's brick is the one core owns", () => {
  assert.equal(DEFAULT_BRICK_FINE_RESOLUTION, BRICK_FINE_CELLS);
  assert.equal(BRICK_FINE_RESOLUTION, BRICK_FINE_CELLS);
  assert.equal(ADVANCE_BRICK_FINE, BRICK_FINE_CELLS);
});

test("the brick is a power of two the region ladder can land on", () => {
  assert.equal(BRICK_FINE_CELLS, 8);
  assert.equal(Math.log2(BRICK_FINE_CELLS) % 1, 0);
});
