import assert from "node:assert/strict";
import test from "node:test";
import { sliceCellOwnerAt, sliceCellOwnerByScan,
  type SliceIndexedTopology } from "./slice-cell-index";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";

/**
 * The owner image replaces a scan that presentation and the stage stencils ran
 * once per fine cell per published lattice. It is only allowed to be faster,
 * so every case here compares it with the scan it replaced rather than with an
 * independently authored expectation.
 */

function assertAgreesEverywhere(topology: SliceIndexedTopology,
  nx: number, ny: number, label: string): void {
  for (let y = -2; y <= ny + 1; y += 1) for (let x = -2; x <= nx + 1; x += 1) {
    for (const [dx, dy] of [[0, 0], [0.5, 0.5], [0.25, 0.75]] as const) {
      const px = x + dx, py = y + dy;
      assert.equal(sliceCellOwnerAt(topology, px, py),
        sliceCellOwnerByScan(topology, px, py),
        `${label}: owner disagreed at ${px},${py}`);
    }
  }
}

test("the owner image answers a production topology exactly as the scan did", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("water-box-dam-break"));
  assertAgreesEverywhere(slice.numericalTopology, slice.nx, slice.ny, "generation zero");
  // A rerung generation publishes a new topology object with coarse cells whose
  // centres sit on whole fine coordinates; those are the queries a naive
  // floor-based image gets wrong.
  for (let frame = 0; frame < 4; frame += 1) {
    advanceSlice(slice, 32);
    assertAgreesEverywhere(slice.numericalTopology, slice.nx, slice.ny, `frame ${frame + 1}`);
  }
});

test("mixed-rung cells own every fine coordinate under them", () => {
  const topology: SliceIndexedTopology = {
    dimensions: [8, 8],
    cells: [
      { id: 0, minimum: [0, 0], maximum: [4, 4] },
      { id: 1, minimum: [4, 0], maximum: [6, 2] },
      { id: 2, minimum: [6, 0], maximum: [7, 1] },
    ],
  };
  assert.equal(sliceCellOwnerAt(topology, 2, 2), 0);
  assert.equal(sliceCellOwnerAt(topology, 3.999, 3.999), 0);
  assert.equal(sliceCellOwnerAt(topology, 4, 0), 1);
  assert.equal(sliceCellOwnerAt(topology, 5.5, 1.5), 1);
  assert.equal(sliceCellOwnerAt(topology, 6.5, 0.5), 2);
  // Uncovered support, and every coordinate outside the domain, stays -1.
  assert.equal(sliceCellOwnerAt(topology, 7.5, 0.5), -1);
  assert.equal(sliceCellOwnerAt(topology, 0.5, 7.5), -1);
  assert.equal(sliceCellOwnerAt(topology, -0.5, 0.5), -1);
  assert.equal(sliceCellOwnerAt(topology, 8.5, 0.5), -1);
  assertAgreesEverywhere(topology, 8, 8, "mixed rungs");
});

test("overlapping support resolves to the first cell, as the scan did", () => {
  const topology: SliceIndexedTopology = {
    dimensions: [4, 4],
    cells: [
      { id: 7, minimum: [0, 0], maximum: [4, 4] },
      { id: 9, minimum: [1, 1], maximum: [3, 3] },
    ],
  };
  assert.equal(sliceCellOwnerAt(topology, 2.5, 2.5), 7);
  assertAgreesEverywhere(topology, 4, 4, "overlap");
});

test("a topology the image cannot represent falls back to the scan", () => {
  const fractional: SliceIndexedTopology = {
    dimensions: [4, 4],
    cells: [{ id: 3, minimum: [0.5, 0.5], maximum: [2.5, 2.5] }],
  };
  assert.equal(sliceCellOwnerAt(fractional, 1, 1), 3);
  assert.equal(sliceCellOwnerAt(fractional, 0.4, 1), -1);
  assertAgreesEverywhere(fractional, 4, 4, "fractional bounds");

  const empty: SliceIndexedTopology = { dimensions: [0, 0], cells: [] };
  assert.equal(sliceCellOwnerAt(empty, 0, 0), -1);
});
