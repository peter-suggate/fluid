import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES,
  SPARSE_CM12_RESIDENT_STAGES,
} from "../webgpu-sparse-cm12-resident";
import {
  ADVANCE_STAGE_ORDER, ADVANCE_WORK, ADVANCE_WORK_SCENES,
  advanceCosts, advanceStageCost, advanceWorkModel,
} from "./advance-work";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import { extendSliceVelocity } from "./slice-stage-numerics";
import {
  advanceSlice, createAdvanceSlice, transitionAdvanceSliceTopology, type SliceStageId,
} from "./slice-solver";

const inputs = (over: Partial<Parameters<typeof advanceWorkModel>[0]> = {}) =>
  advanceWorkModel({
    scene: ADVANCE_WORK_SCENES.mini32, pressureIterations: 64, cfl: 1,
    limiterPasses: 8, churn: 0.06, markers: 65_536, ...over,
  });

test("the work table covers the resident stage ABI, in encode order", () => {
  assert.deepEqual([...ADVANCE_STAGE_ORDER], [...SPARSE_CM12_RESIDENT_STAGES]);
  for (const stage of ADVANCE_STAGE_ORDER) {
    const declared = SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage] as readonly string[];
    const modelled = ADVANCE_WORK[stage].seams
      .flatMap(seam => (seam.id === null ? [] : [seam.id as string]));
    assert.deepEqual(modelled, [...declared]);
  }
});

test("every kernel the work table prices is one the encoder dispatches", () => {
  const source = readFileSync(new URL(
    "../webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
  const missing: string[] = [];
  for (const stage of ADVANCE_STAGE_ORDER) for (const seam of ADVANCE_WORK[stage].seams) {
    for (const kernel of seam.kernels) {
      if (kernel.isCopy) continue;
      const found = kernel.host ? source.includes(kernel.name) : source.includes(`"${kernel.name}"`);
      if (!found) missing.push(`${stage}/${kernel.name}`);
    }
  }
  assert.deepEqual(missing, []);
});

test("a capability a scene lacks encodes no dispatches for it", () => {
  const withInflow = advanceStageCost(
    ADVANCE_WORK["body-forces"], inputs({ scene: ADVANCE_WORK_SCENES.dam }));
  const without = advanceStageCost(ADVANCE_WORK["body-forces"], inputs());
  assert.ok(withInflow.dispatches > without.dispatches);
  assert.equal(without.dispatches, 3);
});

test("the pressure solve keeps its tail encoded past the residual guard", () => {
  const budget = 64;
  const cost = advanceStageCost(
    ADVANCE_WORK["pressure-solve"], inputs({ pressureIterations: budget }));
  assert.equal(cost.dispatches, budget * 3 + (budget / 8 - 1) * 8 + 3);
});

test("transport is the frame's largest stage at mini32 scale", () => {
  const costs = advanceCosts(inputs({ cfl: 1.5 }));
  const largest = costs.indexOf(costs.reduce((a, b) => b.workgroups > a.workgroups ? b : a));
  assert.equal(ADVANCE_STAGE_ORDER[largest], "conservative-transport");
});

function massAndY(slice: ReturnType<typeof createAdvanceSlice>): readonly [number, number] {
  let amount = 0, moment = 0;
  for (const cell of slice.topology.accepted.cells) {
    const volume = slice.fields.density[cell.id]! * cell.volumeFineCells;
    amount += volume;
    moment += volume * cell.centerFine[1];
  }
  return [amount, moment / amount];
}

test("the production coarse translation fixture advances on its native mixed-rung graph", () => {
  const seed = productionSceneSliceSeedById("coarse-surface-translation");
  const slice = createAdvanceSlice(seed);
  assert.deepEqual([...new Set(slice.topology.accepted.bricks
    .filter(brick => brick.active !== false).map(brick => brick.resolution))].sort(), [1, 2]);
  const [beforeMass, beforeY] = massAndY(slice);
  const stages: SliceStageId[] = [];
  advanceSlice(slice, { pressureIterations: 8, onStageComplete: stage => stages.push(stage) });
  const [afterMass, afterY] = massAndY(slice);
  assert.equal(slice.fault, null);
  assert.equal(slice.microsteps, 1);
  assert.ok(Math.abs(afterMass - beforeMass) <= 2e-6);
  const expectedTravelFine = seed.velocityY[0]! * seed.dt / seed.viewport.sourceCellSize;
  assert.ok(Math.abs((beforeY - afterY) - expectedTravelFine) <= 2e-7);
  assert.deepEqual(stages, [
    "transport-velocity-extension", "face-preparation", "body-forces",
    "pressure-topology", "pressure-rhs", "pressure-solve", "velocity-projection",
    "conservative-transport", "tracer-advection", "scalar-publication",
    "activity-measurement", "resolution-planning", "candidate-transfer", "brick-retirement",
    "presentation-publication",
  ]);
});

test("the production sub-isovalue VEX failure is retained", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  slice.fields.density.fill(Math.fround(0.49));
  slice.fields.cellVelocity.fill(Math.fround(3));
  extendSliceVelocity(slice.numericalTopology, slice.fields, 8);
  assert.ok(slice.fields.extensionDepth.every(depth => depth === 255));
  assert.ok(slice.fields.cellVelocity.every(velocity => velocity === 0));
});

test("a production mixed-rung fixture rerungs through candidate field authority", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const before = massAndY(slice);
  const generation = slice.topology.accepted.generation;
  const candidate = slice.topology.accepted.bricks.map(brick => ({ ...brick,
    resolution: brick.active === false ? brick.resolution : 2 as const,
    density: undefined, gamma: undefined,
  }));
  assert.equal(transitionAdvanceSliceTopology(slice, candidate), true);
  const after = massAndY(slice);
  assert.equal(slice.topology.accepted.generation, generation + 1);
  assert.ok(slice.topology.accepted.bricks
    .filter(brick => brick.active !== false).every(brick => brick.resolution === 2));
  assert.ok(Math.abs(after[0] - before[0]) <= 2e-6);
  assert.ok(Math.abs(after[1] - before[1]) <= 2e-7);
});

test("water-box automatic rerung replaces stale rung payload through candidate transfer", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("water-box-dam-break"));
  const generation = slice.topology.accepted.generation;
  assert.equal(slice.topology.accepted.brickByKey.get(8)?.resolution, 4);
  assert.equal(slice.topology.accepted.brickByKey.get(11)?.resolution, 4);
  advanceSlice(slice, { pressureIterations: 8 });
  advanceSlice(slice, { pressureIterations: 8 });
  assert.equal(slice.fault, null);
  assert.equal(slice.topology.accepted.generation, generation + 1);
  assert.equal(slice.resolutionReceipt?.promotedBrickCount, 1);
  assert.equal(slice.resolutionReceipt?.demotedBrickCount, 0);
  const promoted = slice.resolutionReceipt?.bricks.find(record => record.brickKey === 8);
  assert.deepEqual([promoted?.acceptedResolution, promoted?.requestedResolution,
    promoted?.scheduledResolution], [4, 8, 8]);
  assert.equal(slice.topology.accepted.brickByKey.get(8)?.resolution, 8);
  assert.equal(slice.topology.accepted.brickByKey.get(11)?.resolution, 4);
  assert.equal(slice.runtimeAuthority.accepted.topology, slice.topology.accepted);
  assert.equal(slice.runtimeAuthority.receipt.acceptedGeneration, generation + 1);
  assert.equal(slice.runtimeAuthority.acceptedSlot, 1);
});
