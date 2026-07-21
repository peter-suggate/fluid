import assert from "node:assert/strict";
import test from "node:test";
import {
  buildSparseFineLevelSetSummaryHierarchy,
  classifyFineLevelSetSummary,
  summarizeFineLevelSetBrick,
} from "../lib/octree-fine-levelset-summary";
import {
  FINE_LEVELSET_SAMPLE_FLAGS,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
} from "../lib/octree-fine-levelset-bricks";

const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [8, 8, 8],
  finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });

function brick(x: number, y: number, z: number, value: number) {
  const count = plan.samplesPerBrick;
  return summarizeFineLevelSetBrick(plan, packFineLevelSetBrickKey(plan, [x, y, z]), 3,
    new Float32Array(count).fill(value),
    new Uint32Array(count).fill(FINE_LEVELSET_SAMPLE_FLAGS.valid | FINE_LEVELSET_SAMPLE_FLAGS.known));
}

test("complete metric summaries prove conservative refine and coarse decisions", () => {
  assert.equal(classifyFineLevelSetSummary(brick(0, 0, 0, 0.25), {
    generation: 3, refinementDistance: 0.5, samplingRadius: 0.2,
  }), "refine");
  assert.equal(classifyFineLevelSetSummary(brick(0, 0, 0, 4), {
    generation: 3, refinementDistance: 0.5, samplingRadius: 0.2,
  }), "coarse");
});

test("stale and incomplete summaries are inconclusive and cannot coarsen", () => {
  const complete = brick(0, 0, 0, 4);
  assert.equal(classifyFineLevelSetSummary(complete, {
    generation: 4, refinementDistance: 0.5, samplingRadius: 0.2,
  }), "inconclusive");
  const flags = new Uint32Array(plan.samplesPerBrick).fill(FINE_LEVELSET_SAMPLE_FLAGS.valid | FINE_LEVELSET_SAMPLE_FLAGS.known);
  flags[0] = 0;
  const incomplete = summarizeFineLevelSetBrick(plan, 0, 3, new Float32Array(plan.samplesPerBrick).fill(4), flags);
  assert.equal(classifyFineLevelSetSummary(incomplete, {
    generation: 3, refinementDistance: 0.5, samplingRadius: 0.2,
  }), "inconclusive");
});

test("sparse hierarchy materializes resident ancestors only and exposes missing children", () => {
  const partial = buildSparseFineLevelSetSummaryHierarchy([brick(0, 0, 0, 4), brick(1, 0, 0, 5)], 1);
  assert.equal(partial.size, 3);
  const parent = partial.get("1:0,0,0"); assert.ok(parent);
  assert.equal(parent.childMask, 0b11);
  assert.equal(classifyFineLevelSetSummary(parent, {
    generation: 3, refinementDistance: 0.5, samplingRadius: 0.2,
  }), "inconclusive");
});
