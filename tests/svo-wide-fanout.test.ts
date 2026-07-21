import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_WIDE_INVALID_INDEX,
  SVO_WIDE_MICRO_MIP_WORDS,
  planSvoWideFanout,
  svoWideSlotCoordinate,
  traverseSvoWideFanout,
  type SvoWideTerminalInput,
} from "../lib/svo-wide-fanout";

const mixed: readonly SvoWideTerminalInput[] = [
  { sourceNodeIndex: 10, sourceLeafIndex: 1, level: 1, coordinate: [0, 0, 0], solidOpacity: 1 },
  { sourceNodeIndex: 20, sourceLeafIndex: 2, level: 2, coordinate: [3, 0, 0], solidOpacity: 0.5 },
  { sourceNodeIndex: 30, level: 3, coordinate: [4, 0, 0], solidOpacity: 0.25, fluidFraction: 0.5 },
  { sourceNodeIndex: 40, sourceLeafIndex: 4, level: 4, coordinate: [11, 0, 0], solidOpacity: 0.75 },
];

test("4^3 planner collapses two binary levels and preserves mixed-level terminals", () => {
  const plan = planSvoWideFanout({ sourceGeneration: 7, generation: 8, maximumDepth: 4, terminals: mixed });
  assert.equal(plan.pages.length, 2);
  assert.deepEqual(plan.pages.map((page) => [page.level, page.coordinate]), [[0, [0, 0, 0]], [2, [2, 0, 0]]]);
  assert.equal(plan.terminalSlotCount, 18, "each odd-level terminal occupies eight slots");
  assert.equal(plan.descriptorCount, 19, "the root also contains one child-page descriptor");

  const root = plan.pages[0];
  const child = plan.pages[1];
  assert.equal(root.descriptors.filter((entry) => entry.kind === "terminal" && entry.sourceNodeIndex === 10).length, 8);
  assert.equal(root.descriptors.find((entry) => entry.kind === "page")?.slot, 2);
  assert.equal(child.descriptors.filter((entry) => entry.kind === "terminal" && entry.sourceNodeIndex === 30).length, 8);
  assert.equal(child.descriptors.find((entry) => entry.kind === "terminal" && entry.sourceNodeIndex === 40)?.slot, 3);
  assert.equal(mixed[2].sourceLeafIndex, undefined);
  const structural = child.descriptors.find((entry) => entry.kind === "terminal" && entry.sourceNodeIndex === 30);
  assert.equal(structural?.kind === "terminal" ? structural.sourceLeafIndex : 0, SVO_WIDE_INVALID_INDEX);
});
test("planner output is deterministic under terminal input permutation", () => {
  const first = planSvoWideFanout({ sourceGeneration: 7, generation: 8, maximumDepth: 4, terminals: mixed });
  const second = planSvoWideFanout({ sourceGeneration: 7, generation: 8, maximumDepth: 4, terminals: [...mixed].reverse() });
  assert.deepEqual(second, first);
});

test("page-local opacity micro-mips retain means and conservative maxima", () => {
  const plan = planSvoWideFanout({ sourceGeneration: 1, generation: 1, maximumDepth: 2, terminals: [
    { sourceNodeIndex: 1, level: 2, coordinate: [0, 0, 0], solidOpacity: 1, fluidFraction: 0.25 },
    { sourceNodeIndex: 2, level: 2, coordinate: [1, 0, 0], solidOpacity: 0.5, fluidFraction: 1 },
  ] });
  const mips = plan.pages[0].microMips;
  assert.equal(mips.length, SVO_WIDE_MICRO_MIP_WORDS);
  assert.deepEqual(mips[0], { solidMean: 1, solidMaximum: 1, fluidMean: 0.25, fluidMaximum: 0.25 });
  assert.deepEqual(mips[1], { solidMean: 0.5, solidMaximum: 0.5, fluidMean: 1, fluidMaximum: 1 });
  assert.deepEqual(mips[64], { solidMean: 1.5 / 8, solidMaximum: 1, fluidMean: 1.25 / 8, fluidMaximum: 1 });
  assert.deepEqual(mips[72], { solidMean: 1.5 / 64, solidMaximum: 1, fluidMean: 1.25 / 64, fluidMaximum: 1 });
});

test("canonical terminal overlap and invalid fields are rejected before publication", () => {
  assert.throws(() => planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 3, terminals: [
    { sourceNodeIndex: 1, level: 1, coordinate: [0, 0, 0] },
    { sourceNodeIndex: 2, level: 3, coordinate: [1, 1, 1] },
  ] }), /must not overlap/);
  assert.throws(() => planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 2, terminals: [
    { sourceNodeIndex: 1, level: 2, coordinate: [4, 0, 0] },
  ] }), /outside its canonical level/);
  assert.throws(() => planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 2, terminals: [
    { sourceNodeIndex: 1, level: 2, coordinate: [0, 0, 0], solidOpacity: 1.1 },
  ] }), /Solid opacity/);
});

test("near-to-far traversal oracle returns canonical terminal bounds from either direction", () => {
  const plan = planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 2, terminals: [
    { sourceNodeIndex: 10, sourceLeafIndex: 10, level: 1, coordinate: [0, 0, 0] },
    { sourceNodeIndex: 30, sourceLeafIndex: 30, level: 2, coordinate: [3, 0, 0] },
  ] });
  const mapping = { origin: [0, 0, 0] as const, cellSize: [1, 1, 1] as const, brickSize: 4 as const, maximumDepth: 2 };
  const forward = traverseSvoWideFanout({ origin: [-1, 2, 2], direction: [1, 0, 0] }, plan, mapping);
  const backward = traverseSvoWideFanout({ origin: [17, 2, 2], direction: [-1, 0, 0] }, plan, mapping);
  assert.equal(forward.status, "hit");
  assert.equal(backward.status, "hit");
  if (forward.status === "hit") {
    assert.equal(forward.sourceNodeIndex, 10);
    assert.equal(forward.sourceLevel, 1);
    assert.deepEqual(forward.coordinate, [0, 0, 0]);
    assert.deepEqual([forward.tEnter, forward.tExit], [1, 9]);
  }
  if (backward.status === "hit") {
    assert.equal(backward.sourceNodeIndex, 30);
    assert.deepEqual(backward.coordinate, [3, 0, 0]);
    assert.deepEqual([backward.tEnter, backward.tExit], [1, 5]);
  }
});

test("shared-face rays retain canonical lower-slot tie breaking in both x directions", () => {
  const plan = planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 2, terminals: [
    { sourceNodeIndex: 10, sourceLeafIndex: 10, level: 2, coordinate: [0, 0, 0] },
    { sourceNodeIndex: 20, sourceLeafIndex: 20, level: 2, coordinate: [0, 1, 0] },
  ] });
  const mapping = { origin: [0, 0, 0] as const, cellSize: [1, 1, 1] as const, brickSize: 4 as const, maximumDepth: 2 };
  for (const ray of [
    { origin: [-1, 4, 2] as const, direction: [1, 0, 0] as const },
    { origin: [17, 4, 2] as const, direction: [-1, 0, 0] as const },
  ]) {
    const result = traverseSvoWideFanout(ray, plan, mapping);
    assert.equal(result.status, "hit");
    if (result.status === "hit") assert.deepEqual(result.coordinate, [0, 0, 0]);
  }
});

test("traversal reports miss and bounded-work failures distinctly", () => {
  const plan = planSvoWideFanout({ sourceGeneration: 1, generation: 2, maximumDepth: 4, terminals: [
    { sourceNodeIndex: 1, level: 4, coordinate: [0, 0, 0] },
  ] });
  const mapping = { origin: [0, 0, 0] as const, cellSize: [1, 1, 1] as const, brickSize: 4 as const, maximumDepth: 4 };
  assert.equal(traverseSvoWideFanout({ origin: [-1, 100, 0], direction: [1, 0, 0] }, plan, mapping).status, "miss");
  assert.equal(traverseSvoWideFanout(
    { origin: [-1, 1, 1], direction: [1, 0, 0] }, plan, mapping, { maximumPageVisits: 1 },
  ).status, "work-exhausted");
  assert.equal(traverseSvoWideFanout(
    { origin: [-1, 1, 1], direction: [1, 0, 0] }, plan, mapping, { stackCapacity: 1 },
  ).status, "hit", "a single-child hierarchy fits the minimum stack");
});

test("slot addressing is x-major over a 4x4x4 page", () => {
  assert.deepEqual(svoWideSlotCoordinate(0), [0, 0, 0]);
  assert.deepEqual(svoWideSlotCoordinate(27), [3, 2, 1]);
  assert.deepEqual(svoWideSlotCoordinate(63), [3, 3, 3]);
});
