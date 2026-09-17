import assert from "node:assert/strict";
import test from "node:test";

// The registry is installed by side effect; `getScenePreset` reaches it.
import "../lib/methods";

import {
  refinementRegionsToQuery, withRefinementRegionsFromQuery,
} from "../lib/core/editor-refinement-region";
import { refinementRegionLattice } from "../lib/core/refinement-regions";
import { getScenePreset } from "../lib/core/scenes";
import { regionsToQuery } from "../lib/features/refinement-region/persistence";
import { labRegionSpace } from "../advance-lab/lab-region-space";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";

/**
 * One `regions=` value, read and written by both hosts.
 *
 * `lib/features/refinement-region/verification/persistence.test.ts` holds the
 * encoding itself against fabricated spaces. This file is the claim that
 * matters to a reader holding a link: the *shipped* adapters — metres and a
 * scene document on one side, finest cells on a Wasm slice on the other —
 * agree, and the value a studio link has carried since before the package
 * existed still means what it meant.
 */

/** The scene a preset builds, as the editor's stores hold it. */
function sceneOf(id: string) {
  const preset = getScenePreset(id);
  assert.ok(preset, `no preset ${id}`);
  return preset.create();
}

test("the shipped studio link still parses and re-writes to itself", () => {
  // `tests/compare-model.test.ts:548` has carried this literal across every
  // refactor of the region editor. It is a whole-domain box at the two-cell
  // rung; the package reads it against the lattice rather than the container,
  // and this is where that claim meets a real scene.
  const scene = sceneOf("water-box-dam-break");
  const applied = withRefinementRegionsFromQuery(scene, "0_0_0_100_100_100_2");
  assert.equal(refinementRegionsToQuery(applied), "0_0_0_100_100_100_2");

  const regions = applied.fluid.refinementRegions ?? [];
  assert.equal(regions.length, 1);
  const { origin_m, cellSize_m, dimensions } = refinementRegionLattice(scene);
  const region = regions[0]!;
  // The whole container, in metres, to within a thousandth of a cell.
  for (const [axis, index] of [["x", 0], ["y", 1], ["z", 2]] as const) {
    const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9,
      `${axis}: ${a} vs ${b}`);
    near(region.min_m[axis], origin_m[axis]);
    near(region.max_m[axis], origin_m[axis] + cellSize_m[index]! * dimensions[index]!);
  }
  assert.equal(region.minimumCellSize_cells, 2);
});

test("an emptied value clears the scene's regions rather than keeping them", () => {
  const scene = withRefinementRegionsFromQuery(
    sceneOf("water-box-dam-break"), "0_0_0_100_100_100_2");
  assert.equal(refinementRegionsToQuery(withRefinementRegionsFromQuery(scene, "")), "");
});

/** The lab's document for a box in finest cells on an `nx` by `ny` slice. */
function labDoc(nx: number, ny: number, regions: readonly AdvanceRefinementRegion[]) {
  return { nx, ny, regions };
}

test("both hosts write the same percentages for the same box", () => {
  // A scene whose lattice is a power of two on every axis, so a side the reader
  // can actually place is an exact percentage: the wire carries four decimals,
  // which is a hundredth of a percent of the domain rather than a cell address.
  const scene = sceneOf("high-resolution-dam-break");
  const { dimensions } = refinementRegionLattice(scene);
  assert.deepEqual([...dimensions], [128, 128, 128]);

  // The same box, stated to each host in its own frame: x and y in finest
  // cells, and the studio's third axis spanning the depth.
  const studio = refinementRegionsToQuery(
    withRefinementRegionsFromQuery(scene, "25_12.5_0_50_37.5_100_8"));
  const lab = regionsToQuery(labRegionSpace, labDoc(128, 128, [{
    id: "advance-region-1",
    minimumFine: [32, 16],
    maximumFine: [64, 48],
    minimumCellWidth: 8,
  }]));

  assert.equal(lab, "25_12.5_50_37.5_8");
  // The studio's value is the lab's with the depth spliced into the corners:
  // three minima, three maxima, then the same tail.
  assert.equal(studio, "25_12.5_0_50_37.5_100_8");
  const [x0, y0, , x1, y1, ...tail] = studio.split("_");
  assert.equal([x0, y0, x1, y1, ...tail.slice(1)].join("_"), lab);
});

test("the lab's y-flip is a drawing concern, not a wire one", () => {
  // The record — and so the value — is the frame the solver reads a region in,
  // with y running up from the floor. A box against the floor must therefore
  // write `0` for its y minimum on both hosts, not `ny - max`.
  const floorBox = regionsToQuery(labRegionSpace, labDoc(128, 64, [{
    id: "advance-region-1",
    minimumFine: [0, 0],
    maximumFine: [128, 16],
    minimumCellWidth: 8,
  }]));
  assert.equal(floorBox, "0_0_100_25_8");
});
