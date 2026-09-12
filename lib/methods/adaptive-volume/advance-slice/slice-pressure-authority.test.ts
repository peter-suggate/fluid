import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER as PCM_D,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE,
} from "../sparse-cm12-canonical-membership";
import {
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER as PEI_H,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE,
} from "../sparse-cm12-pressure-execution-image";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import {
  createSlicePressureAuthority,
  publishSlicePressureAuthority,
} from "./slice-pressure-authority";
import { prepareSlicePressureTopology, reconstructSliceInterfaces } from
  "./slice-stage-numerics";
import { createAdvanceSlice } from "./slice-solver";

test("PCM1/PCF1/PEI1 publish the slice pressure epoch in stable-cell order", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  reconstructSliceInterfaces(slice.numericalTopology, slice.fields);
  const rows = prepareSlicePressureTopology(slice.numericalTopology, slice.fields);
  const initial = createSlicePressureAuthority(slice.numericalTopology, {
    cells: slice.pressureAuthority.cellCapacity,
    rows: slice.pressureAuthority.rowCapacity,
    bricks: Math.ceil(slice.pressureAuthority.cellCapacity / 64),
  });
  const first = publishSlicePressureAuthority(initial, slice.numericalTopology,
    slice.fields, rows, slice.topology.accepted.generation);
  assert.equal(first.receipt.fault, 0);
  assert.equal(first.receipt.pcmCellGeneration, 1);
  assert.equal(first.executionOrder.length,
    slice.fields.pressureMember.reduce((sum, value) => sum + value, 0));
  const cellHeader = first.pcmLayout.cell.headerBaseWords;
  assert.equal(first.pcmWords[cellHeader + PCM_D.phase],
    SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted);
  assert.equal(first.pcmWords[cellHeader + PCM_D.totalCount], first.executionOrder.length);
  const pei = first.peiLayout.baseWords;
  assert.equal(first.peiWords[pei + PEI_H.phase],
    SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.accepted);
  assert.equal(first.peiWords[pei + PEI_H.topologyGeneration],
    slice.topology.accepted.generation);
  assert.equal(first.peiWords[pei + PEI_H.pressureCellCount], first.executionOrder.length);
  const stable = Array.from(first.executionOrder, dense =>
    slice.numericalTopology.cells[dense]!.stableId!);
  assert.deepEqual(Array.from(first.peiWords.slice(first.peiLayout.pressureCellBaseWords,
    first.peiLayout.pressureCellBaseWords + stable.length)), stable);
});

test("unchanged pressure authority publishes a new generation with empty repair worklists", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  reconstructSliceInterfaces(slice.numericalTopology, slice.fields);
  const firstRows = prepareSlicePressureTopology(slice.numericalTopology, slice.fields);
  const first = publishSlicePressureAuthority(slice.pressureAuthority,
    slice.numericalTopology, slice.fields, firstRows, slice.topology.accepted.generation);
  const secondRows = prepareSlicePressureTopology(slice.numericalTopology, slice.fields);
  const second = publishSlicePressureAuthority(first,
    slice.numericalTopology, slice.fields, secondRows, slice.topology.accepted.generation);
  assert.equal(second.receipt.pcmCellGeneration, first.receipt.pcmCellGeneration + 1);
  assert.equal(second.receipt.pcmRowGeneration, first.receipt.pcmRowGeneration + 1);
  assert.deepEqual(Array.from(second.receipt.dirtyCellLeaves), []);
  assert.deepEqual(Array.from(second.receipt.dirtyRowTiles), []);
  assert.deepEqual(Array.from(second.executionOrder), Array.from(first.executionOrder));
});

test("a capacity overflow fail-closes without replacing accepted pressure authority", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  reconstructSliceInterfaces(slice.numericalTopology, slice.fields);
  const rows = prepareSlicePressureTopology(slice.numericalTopology, slice.fields);
  const undersized = createSlicePressureAuthority(slice.numericalTopology,
    { cells: 1, rows: 1, bricks: 1 });
  const failed = publishSlicePressureAuthority(undersized, slice.numericalTopology,
    slice.fields, rows, slice.topology.accepted.generation);
  assert.equal(failed.receipt.fault, 3);
  assert.equal(failed.receipt.executionGeneration, 0);
  assert.equal(failed.executionOrder.length, 0);
});
