import assert from "node:assert/strict";
import test from "node:test";

import { SPARSE_CM12_FINAL_SCALAR_MASK_HEADER,
  SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC,
  SPARSE_CM12_FINAL_SCALAR_MASK_PHASE } from "../sparse-cm12-final-scalar-packet-masks";
import { createAdvanceSlice } from "./slice-solver";
import { createSliceScalarAuthority, publishSliceScalarAuthority } from "./slice-scalar-authority";

test("FSM1 publishes stable TEI packet bits and exact source-bank facts", () => {
  const slice = createAdvanceSlice(), beforeDensity = slice.fields.density.slice(),
    beforeGamma = slice.fields.gamma.slice();
  slice.fields.characteristicClearance = new Float32Array(slice.fields.density.length);
  const partial = slice.topology.accepted.cells.find(cell => slice.fields.density[cell.id]! > 0
    && slice.fields.density[cell.id]! < 1) ?? slice.topology.accepted.cells[0]!;
  slice.fields.density[partial.id] = Math.fround(slice.fields.density[partial.id]! + 0.125);
  let authority = createSliceScalarAuthority(slice.runtimeAuthority,
    slice.fields.density.length);
  authority = publishSliceScalarAuthority(authority, slice.topology.accepted,
    slice.numericalTopology, slice.fields, slice.runtimeAuthority,
    beforeDensity, beforeGamma, 7, false);
  const h = SPARSE_CM12_FINAL_SCALAR_MASK_HEADER;
  assert.equal(authority.words[h.magic], SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC);
  assert.equal(authority.words[h.phase], SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.published);
  assert.equal(authority.receipt.generation, 7);
  assert.equal(authority.receipt.changedCellCount, 1);
  assert.ok(authority.receipt.nonexactCellCount >= 1);
});

test("FSM1 bulk certificate mirrors current values into the dead scalar bank", () => {
  const slice = createAdvanceSlice();
  slice.fields.density.fill(1); slice.fields.gamma.fill(1);
  slice.fields.capacity.fill(1);
  slice.fields.characteristicClearance = new Float32Array(slice.fields.density.length).fill(0.02);
  const beforeDensity = slice.fields.density.slice(), beforeGamma = slice.fields.gamma.slice();
  const authority = publishSliceScalarAuthority(
    createSliceScalarAuthority(slice.runtimeAuthority, slice.fields.density.length),
    slice.topology.accepted, slice.numericalTopology, slice.fields, slice.runtimeAuthority,
    beforeDensity, beforeGamma, 1, false);
  assert.equal(authority.receipt.bulkCellCount, slice.fields.density.length);
  assert.deepEqual(authority.sourceDensity, slice.fields.density);
  assert.deepEqual(authority.sourceGamma, slice.fields.gamma);
});

