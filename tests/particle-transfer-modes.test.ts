import assert from "node:assert/strict";
import test from "node:test";
import { APIC_PARAMS, particleTransferOptions } from "../lib/methods/particle/parameters";

test("particle transfer defaults to APIC and exposes live PIC/FLIP controls", () => {
  assert.deepEqual(particleTransferOptions(), { transferMode: 0, flipRatio: 0.95 });
  assert.equal(particleTransferOptions({ transferMode: "pic" }).transferMode, 1);
  assert.equal(particleTransferOptions({ transferMode: "flip" }).transferMode, 2);
  assert.equal(particleTransferOptions({ transferMode: "stale-mode" }).transferMode, 0);
  for (const key of ["transferMode", "flipRatio"]) assert.equal(APIC_PARAMS.find(p => p.key === key)?.update, "runtime");
});

test("FLIP blend accepts both physical endpoints and clamps invalid values", () => {
  for (const [value, expected] of [[0, 0], [1, 1], [-1, 0], [2, 1], [NaN, 0.95], [Infinity, 0.95]]) {
    assert.equal(particleTransferOptions({ flipRatio: value }).flipRatio, expected);
  }
});
