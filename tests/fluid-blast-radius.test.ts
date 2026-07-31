import assert from "node:assert/strict";
import test from "node:test";
import {
  blastRadiusLevelDimensions,
  blastRadiusLevelsToSingleCell,
  growBlastRadius,
  planBlastRadiusSchedule,
  summarizeBlastRadius,
  type BlastRadiusVec3,
} from "../lib/fluid-blast-radius";

/** The mini dam: a 16^3 finest grid over a four-level SPGrid pyramid. */
const MINI: BlastRadiusVec3 = [16, 16, 16];

test("level dimensions halve with the V-cycle's own rounding", () => {
  assert.deepEqual(blastRadiusLevelDimensions([16, 16, 16], 0), [16, 16, 16]);
  assert.deepEqual(blastRadiusLevelDimensions([16, 16, 16], 3), [2, 2, 2]);
  // Odd extents round up, matching `(dims + (1 << l) - 1) >> l`.
  assert.deepEqual(blastRadiusLevelDimensions([17, 5, 1], 1), [9, 3, 1]);
  assert.deepEqual(blastRadiusLevelDimensions([17, 5, 1], 4), [2, 1, 1]);
});

test("a V-cycle schedule is symmetric around its coarsest solve", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 4, smoothsPerLevel: 1 });
  assert.deepEqual(schedule.stages.map((stage) => stage.kind), [
    "operator",
    "smooth", "restrict",
    "smooth", "restrict",
    "smooth", "restrict",
    "coarse-solve",
    "prolong", "smooth",
    "prolong", "smooth",
    "prolong", "smooth",
  ]);
  assert.ok(schedule.stages.every((stage) => stage.iteration === 0));
});

test("outer iterations repeat the whole cycle", () => {
  const one = planBlastRadiusSchedule({ outerIterations: 1, levels: 3, smoothsPerLevel: 2 });
  const four = planBlastRadiusSchedule({ outerIterations: 4, levels: 3, smoothsPerLevel: 2 });
  assert.equal(four.stages.length, one.stages.length * 4);
  assert.equal(four.stages.at(-1)?.iteration, 3);
});

test("the schedule rejects a degenerate hierarchy rather than modelling one", () => {
  assert.throws(() => planBlastRadiusSchedule({ outerIterations: 0, levels: 4, smoothsPerLevel: 1 }), RangeError);
  assert.throws(() => planBlastRadiusSchedule({ outerIterations: 1, levels: 0, smoothsPerLevel: 1 }), RangeError);
});

test("the cone starts as the single selected cell", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 4, smoothsPerLevel: 1 });
  const frontiers = growBlastRadius({ dimensions: MINI, schedule, cell: [8, 8, 8] });
  assert.equal(frontiers[0].stageIndex, -1);
  assert.equal(frontiers[0].cells, 1);
  assert.deepEqual([...frontiers[0].boxes[0].lo], [8, 8, 8]);
  // Coarse levels are untouched until restriction carries the cone into them.
  assert.ok(frontiers[0].boxes.slice(1).every((box) => box.empty));
});

test("one smoother sweep grows the cone by exactly one cell in each direction", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 4, smoothsPerLevel: 1 });
  const frontiers = growBlastRadius({ dimensions: MINI, schedule, cell: [8, 8, 8] });
  // Stage 0 is the operator application: an 18-point stencil, Chebyshev radius one.
  assert.equal(frontiers[1].cells, 27);
  assert.deepEqual([...frontiers[1].boxes[0].lo], [7, 7, 7]);
  assert.deepEqual([...frontiers[1].boxes[0].hi], [9, 9, 9]);
});

test("the cone clamps at the domain wall instead of growing outside it", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 2, smoothsPerLevel: 1 });
  const frontiers = growBlastRadius({ dimensions: [4, 4, 4], schedule, cell: [0, 0, 0] });
  assert.deepEqual([...frontiers[1].boxes[0].lo], [0, 0, 0]);
  assert.equal(frontiers[1].cells, 8);
});

test("the coarsest solve only fires once the cone has reached that level", () => {
  // Two levels with no restriction ahead of the solve would leave level 1
  // empty; saturating it anyway would invent influence from nothing.
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 2, smoothsPerLevel: 1 });
  const frontiers = growBlastRadius({ dimensions: MINI, schedule, cell: [8, 8, 8] });
  const beforeRestrict = frontiers[1];
  assert.ok(beforeRestrict.boxes[1].empty);
  const solved = frontiers.find((frontier) => frontier.stage?.kind === "coarse-solve");
  assert.ok(solved);
  assert.equal(solved.boxes[1].empty, false);
  // Level 1 of a 16^3 grid is 8^3, and the direct solve couples all of it.
  assert.equal(
    (solved.boxes[1].hi[0] - solved.boxes[1].lo[0] + 1)
    * (solved.boxes[1].hi[1] - solved.boxes[1].lo[1] + 1)
    * (solved.boxes[1].hi[2] - solved.boxes[1].lo[2] + 1),
    8 * 8 * 8,
  );
});

test("the mini dam's cone covers the whole grid inside one V-cycle", () => {
  // This is the finding the view exists to make visible: the coarse level is a
  // shortcut, so a cell's dependency cone stops being local almost immediately.
  const schedule = planBlastRadiusSchedule({ outerIterations: 4, levels: 4, smoothsPerLevel: 1 });
  const frontiers = growBlastRadius({ dimensions: MINI, schedule, cell: [8, 8, 8] });
  const summary = summarizeBlastRadius(frontiers, schedule, MINI);
  assert.equal(summary.totalCells, 4_096);
  assert.equal(summary.iterationToGlobal, 0);
  assert.equal(summary.shareAfterFirstIteration, 1);
  assert.ok(summary.stagesToGlobal !== undefined && summary.stagesToGlobal <= schedule.stages.length / 4,
    "the cone must go global within the first outer iteration");
});

test("a corner cell reaches the whole grid just as fast as a centre cell", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 4, levels: 4, smoothsPerLevel: 1 });
  const centre = summarizeBlastRadius(
    growBlastRadius({ dimensions: MINI, schedule, cell: [8, 8, 8] }), schedule, MINI);
  const corner = summarizeBlastRadius(
    growBlastRadius({ dimensions: MINI, schedule, cell: [0, 0, 0] }), schedule, MINI);
  assert.equal(corner.stagesToGlobal, centre.stagesToGlobal);
});

test("the hierarchy depth is fixed by the exact one-cell bottom", () => {
  // The V-cycle refuses a maximumLevels that would truncate its single-cell
  // bottom, so depth follows the domain rather than a tuning knob.
  assert.equal(blastRadiusLevelsToSingleCell([16, 16, 16]), 5);
  assert.equal(blastRadiusLevelsToSingleCell([1, 1, 1]), 1);
  assert.equal(blastRadiusLevelsToSingleCell([320, 96, 80]), 10);
  const deepest = blastRadiusLevelsToSingleCell([320, 96, 80]);
  assert.deepEqual(blastRadiusLevelDimensions([320, 96, 80], deepest - 1), [1, 1, 1]);
});

test("with no hierarchy the cone grows one cell per sweep and stays local", () => {
  // The smoother alone is the baseline the coarse grid is measured against:
  // four sweeps reach a radius of four, nowhere near a 64^3 domain.
  const schedule = planBlastRadiusSchedule({ outerIterations: 4, levels: 1, smoothsPerLevel: 1 });
  // A one-level schedule's only stages are the operator and the bottom solve,
  // and at level 0 that bottom is the whole grid, so exercise the smoother
  // reach directly instead: eight sweeps from the centre reach radius eight.
  const smoothing = {
    ...schedule,
    stages: schedule.stages.filter((stage) => stage.kind === "operator"),
  };
  const frontiers = growBlastRadius({ dimensions: [64, 64, 64], schedule: smoothing, cell: [32, 32, 32] });
  const summary = summarizeBlastRadius(frontiers, smoothing, [64, 64, 64]);
  assert.equal(summary.stagesToGlobal, undefined);
  assert.equal(frontiers.at(-1)?.cells, 9 ** 3);
});

test("every V-cycle makes the cone global, at any domain size", () => {
  // The one-cell bottom is prolonged back down doubling at each level, so this
  // is exact rather than a modelling artifact of the box representation.
  for (const dimensions of [[16, 16, 16], [64, 64, 64], [320, 96, 80]] as const) {
    const levels = blastRadiusLevelsToSingleCell(dimensions);
    const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels, smoothsPerLevel: 1 });
    const frontiers = growBlastRadius({ dimensions, schedule, cell: [0, 0, 0] });
    const summary = summarizeBlastRadius(frontiers, schedule, dimensions);
    assert.equal(summary.shareAfterFirstIteration, 1, `${dimensions.join("x")} did not go global`);
  }
});

test("fine-grid sweeps count the level-0 work the fast cone costs", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 4, levels: 4, smoothsPerLevel: 2 });
  const summary = summarizeBlastRadius(
    growBlastRadius({ dimensions: MINI, schedule, cell: [1, 1, 1] }), schedule, MINI);
  // Per iteration: one operator, two pre-smooths and two post-smooths at level 0.
  assert.equal(summary.fineGridSweeps, 4 * 5);
});

test("a wider restriction stencil only ever grows the cone", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 4, smoothsPerLevel: 1 });
  const narrow = growBlastRadius({ dimensions: [64, 64, 64], schedule, cell: [32, 32, 32] });
  const wide = growBlastRadius({
    dimensions: [64, 64, 64], schedule, cell: [32, 32, 32],
    restrictionRadius: 1, prolongationRadius: 1,
  });
  assert.ok(wide.at(-1)!.cells >= narrow.at(-1)!.cells);
});

test("a cell outside the domain is rejected", () => {
  const schedule = planBlastRadiusSchedule({ outerIterations: 1, levels: 2, smoothsPerLevel: 1 });
  assert.throws(() => growBlastRadius({ dimensions: MINI, schedule, cell: [16, 0, 0] }), RangeError);
  assert.throws(() => growBlastRadius({ dimensions: MINI, schedule, cell: [-1, 0, 0] }), RangeError);
});
