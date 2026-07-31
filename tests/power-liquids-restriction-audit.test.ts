import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { auditSection5FineRestriction } from "../lib/power-liquids-restriction-audit";

test("the factor-1 no-fine-band variant does not claim the paper's separate-band restriction receipt", () => {
  const executor = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(executor,
    /const hasSeparateFineLevelSetBand = method\.id === "octree"\s*&& values\.globalFineLevelSetFactor !== "1";/);
  assert.match(executor,
    /if \(restrictionAudit\.failure && hasSeparateFineLevelSetBand\s*&& !retainedBackgroundOctree\)/);
});

const restriction = (unacceptedRows: number, rowCount = 363_336) => ({
  contributionCount: rowCount,
  maximumContributionsPerRow: 1,
  flags: 0,
  unacceptedRows,
  rowCount,
  valid: true,
});
const coarse = (correctedRows: number, interfaceRows = 4_073, rowCount = 363_336) => ({
  flags: 0,
  firstErrorRow: 0xffff_ffff,
  rowCount,
  advectedRows: rowCount,
  uniformUpdates: 0,
  transitionUpdates: 0,
  representationPasses: 0,
  correctedRows,
  interfaceRows,
  contributionCount: rowCount,
  generation: 92,
  valid: 0x8000_0000,
});

test("Section 5 accepts a narrow fine band over a separate background octree", () => {
  // Aanjaneya et al. 2017, Section 5
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) uses a high-resolution
  // SPGrid only around the free surface and a separate background octree.
  // The reproduced ocean has 90.62% of coarse rows outside the fine band;
  // that is expected, not a 25%-coverage failure.
  assert.deepEqual(auditSection5FineRestriction(
    restriction(329_252), coarse(34_084)), {
    acceptedRows: 34_084,
    unacceptedRows: 329_252,
    interfaceRows: 4_073,
    correctedRows: 34_084,
  });
});

test("Section 5 restriction rejects a broken correction transaction", () => {
  assert.equal(auditSection5FineRestriction(
    restriction(329_252), coarse(34_083)).failure, "correction-count-mismatch");
  assert.equal(auditSection5FineRestriction(
    restriction(359_336), coarse(4_000, 4_073)).failure, "interface-not-covered");
});
