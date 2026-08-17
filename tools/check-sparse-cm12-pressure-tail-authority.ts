#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  acceptSparseCM12PressureTailGeneration,
  compareSparseCM12PressureTailAuthorityQA,
  createSparseCM12PressureTailCopyPlan,
  assertSparseCM12PressureTailIntegration,
  sparseCM12PressureTailSeams,
  type SparseCM12PressureTailIntegrationStep,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-tail-authority";
import { createSparseCM12PressureSolveAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-solve-authority";

const layout = createSparseCM12PressureSolveAuthorityLayout({
  baseWords: 64, brickCapacity: 500, hierarchyLevelCounts: [100, 16],
});
const plan = createSparseCM12PressureTailCopyPlan(layout);
assert.equal(plan.destinationByteLength, 48);
assert.notEqual(plan.sourceByteOffsets[0], plan.sourceByteOffsets[1]);
assert.deepEqual(sparseCM12PressureTailSeams(4).map(({ bank }) => bank), [0, 1, 0, 1]);

const steps: SparseCM12PressureTailIntegrationStep[] = [];
for (const { bank } of sparseCM12PressureTailSeams(2)) {
  steps.push({ kind: "publish", bank }, { kind: "close-pass" }, { kind: "copy", bank });
  for (const family of ["cell", "wetBrick", "hierarchyNode", "scalar"] as const) {
    steps.push({ kind: "dispatch", bank, family });
  }
  steps.push({ kind: "gate" });
}
assert.doesNotThrow(() => assertSparseCM12PressureTailIntegration(steps));
assert.throws(() => assertSparseCM12PressureTailIntegration([
  { kind: "publish", bank: 0 }, { kind: "copy", bank: 0 }, { kind: "gate" },
]));

const full = [{ acceptedGeneration: 7, publishedGeneration: 7, fault: 0,
  arithmeticActive: true, familyWorkgroups: [17, 23, 5, 1] as const,
  outputWords: [1, 2, 3] }];
assert.equal(compareSparseCM12PressureTailAuthorityQA(full, full).exact, true);
const inactive = [{ ...full[0]!, arithmeticActive: false,
  familyWorkgroups: [0, 0, 0, 0] as const }];
assert.equal(compareSparseCM12PressureTailAuthorityQA(full, inactive).exact, true);
const stale = [{ ...full[0]!, publishedGeneration: 6 }];
assert.equal(compareSparseCM12PressureTailAuthorityQA(full, stale).reason, "generation");
const aliased = [{ ...full[0]!, familyWorkgroups: [17, 23, 0, 1] as const }];
assert.equal(compareSparseCM12PressureTailAuthorityQA(full, aliased).firstMismatchFamily,
  "hierarchyNode");
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: true, acceptedGeneration: 7, publishedGeneration: 7,
  activeBank: 0, fault: 0, predicate: "ordinary", predicateActive: true,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [17, 23, 5, 1],
}).accepted, true);
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: true, acceptedGeneration: 7, publishedGeneration: 0,
  activeBank: 0, fault: 0, predicate: "ordinary", predicateActive: false,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [0, 0, 0, 0],
}).reason, "missing-publication");
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: false, acceptedGeneration: 7, publishedGeneration: 7,
  activeBank: 0, fault: 14, predicate: "ordinary", predicateActive: false,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [17, 0, 0, 0],
}).reason, "fail-open-work");
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: true, acceptedGeneration: 7, publishedGeneration: 7,
  activeBank: 1, fault: 0, predicate: "recovery", predicateActive: true,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [17, 23, 5, 1],
}).accepted, true);
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: true, acceptedGeneration: 7, publishedGeneration: 7,
  activeBank: 1, fault: 0, predicate: "recovery", predicateActive: false,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [17, 0, 0, 0],
}).reason, "predicate-work");
assert.equal(acceptSparseCM12PressureTailGeneration({
  phaseAccepted: true, acceptedGeneration: 7, publishedGeneration: 7,
  activeBank: 1, fault: 0, predicate: "recovery", predicateActive: false,
  expectedFamilyWorkgroups: [17, 23, 5, 1], familyWorkgroups: [0, 0, 0, 0],
}).accepted, true);

// Reuse the PSA1 production/full-oracle Naga checker. PTL1 is deliberately an
// adapter around those exact publishers rather than a second shader ABI.
const naga = spawnSync(process.execPath, ["--import", "tsx",
  "tools/check-sparse-cm12-pressure-solve-authority.ts"], {
  cwd: process.cwd(), encoding: "utf8", env: process.env,
});
if (naga.error) throw naga.error;
if (naga.status !== 0) throw new Error(naga.stderr || naga.stdout);

process.stdout.write(`${JSON.stringify({
  abi: plan.abi, banks: 2, families: 4, bankBytes: plan.destinationByteLength,
  copyIsolated: true, fixedAlternation: true, cpuOracle: "exact",
  receiptPredicates: ["ordinary", "recovery"], predicateWorkgroups: "exact-or-zero",
  naga: "PSA production+construction oracle valid",
}, null, 2)}\n`);
