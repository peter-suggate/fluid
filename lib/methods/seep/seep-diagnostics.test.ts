/** Seep solver (lane S): CPU checks that a pressure solve stopped at the cycle cap is reported as a failure. GPU
 * behaviour (a forced-cap run that throws in the bench) is checked with tools/seep-solver-smoke.ts. */
import assert from "node:assert/strict";
import test from "node:test";
import { pcgCheckKernel } from "./seep-kernels";
import { FAIL } from "./seep-wgsl";

test("seep failure bits are distinct single bits", () => {
  const bits = Object.values(FAIL);
  for (const b of bits) assert.equal(b & (b - 1), 0, `FAIL bit ${b} is not a single bit`);
  assert.equal(new Set(bits).size, bits.length);
});

test("seep pcgCheck latches FAIL.PRESSURE_CAP when it stops above the tolerance", () => {
  const code = pcgCheckKernel.code;
  assert.match(code, new RegExp(`const FAIL_CAP: u32 = ${FAIL.PRESSURE_CAP}u;`));
  // The cap branch (k at MAXC, residual above TOL) both counts the step and latches the failure bit.
  const stop = code.indexOf("if (res <= TOL || k >= MAXC) {");
  assert.ok(stop >= 0, "pcgCheck has no stopping branch");
  const branch = code.slice(stop, code.indexOf("} else {", stop));
  assert.match(branch, /if \(res > TOL\) \{[^}]*atomicAdd\(&ctrl\[C_CAP_HITS\], 1u\);[^}]*atomicOr\(&ctrl\[C_FAIL\], FAIL_CAP\);[^}]*\}/);
});
