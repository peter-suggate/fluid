import assert from "node:assert/strict";
import test from "node:test";

import { createAdvanceSlice } from "./slice-solver";
import { createSliceTracerAuthority, advanceSliceTracerAuthority,
  setSliceTracersEnabled } from "./slice-tracer-authority";

test("tracers retain fixed lanes, seed only rho > 0.5, and advance on enable", () => {
  const slice = createAdvanceSlice();
  slice.fields.density.fill(1);
  slice.fields.cellVelocity.fill(0);
  for (let cell = 0; cell < slice.fields.density.length; cell += 1) {
    slice.fields.cellVelocity[2 * cell] = 1;
  }
  let authority = setSliceTracersEnabled(
    createSliceTracerAuthority([slice.nx, slice.ny], 64), true);
  const initial = authority.state;
  let receipt;
  [authority, receipt] = advanceSliceTracerAuthority(authority,
    slice.numericalTopology, slice.fields, 0.25);
  assert.equal(receipt.seeded, true);
  assert.equal(receipt.liveCount > 0, true);
  assert.equal(authority.state.length, initial.length);
  for (let i = 0; i < authority.lattice.count; i += 1) {
    if (authority.state[4 * i + 3]! < 0.5) continue;
    assert.ok(authority.state[4 * i]! > authority.lattice.originFine[0]);
  }
});

test("a live tracer retires at the resident dry cutoff without compaction", () => {
  const slice = createAdvanceSlice();
  slice.fields.density.fill(1);
  let authority = setSliceTracersEnabled(
    createSliceTracerAuthority([slice.nx, slice.ny], 16), true);
  [authority] = advanceSliceTracerAuthority(authority,
    slice.numericalTopology, slice.fields, 0);
  const count = authority.lattice.count, bytes = authority.state.byteLength;
  slice.fields.density.fill(0);
  let receipt;
  [authority, receipt] = advanceSliceTracerAuthority(authority,
    slice.numericalTopology, slice.fields, 0);
  assert.equal(receipt.retiredCount, count);
  assert.equal(receipt.liveCount, 0);
  assert.equal(authority.state.byteLength, bytes);
});

