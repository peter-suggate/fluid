import assert from "node:assert/strict";
import test from "node:test";
import {
  losassoStepSnapshotDiagnosticSummary,
  losassoStepSnapshotFailures,
  type LosassoStepSnapshotRecord,
} from "../lib/webgpu-octree-losasso-step-snapshot";

const words = (length: number, entries: Readonly<Record<number, number>>) => {
  const result = new Uint32Array(length);
  for (const [index, value] of Object.entries(entries)) result[Number(index)] = value;
  return result;
};

function adaptiveRecord(candidateEpoch: number): LosassoStepSnapshotRecord {
  const acceptedGraph = words(32, {
    0: 17, 1: 1886, 2: 2764, 3: 17, 5: 36, 6: 36, 28: 1240,
  });
  const phiReceipts = words(64, {
    1: 1, 4: 2764, 7: 2764, 14: 1, 15: 1886, 22: 1, 23: 36, 32: 2764,
  });
  const velocityReceipts = words(52, {
    0: 36, 6: 2764, 7: 1, 12: 36, 18: 2764, 19: 1,
  });
  const candidateAuthority = words(8, {
    0: candidateEpoch,
    1: candidateEpoch === 0 ? 0 : 1240,
    2: candidateEpoch === 0 ? 0 : 4306,
    3: candidateEpoch === 0 ? 0 : 1,
    // This is the exact dormant tuple seen in the UI: scratch receipts retain
    // their fail-closed sentinels even though authority epoch zero says there
    // is no candidate transaction to validate.
    4: 0x8000_0004,
  });
  return {
    step: 16,
    surfaceKind: "adaptive",
    authority: words(8, { 0: 17, 1: 1240, 2: 4306, 3: 1 }),
    solver: words(16, { 1: 1, 4: 1240 }),
    fine: new Uint32Array(0),
    coarsePhi: words(16, {
      0: 0x4c50_4849, 1: 17, 2: 1886, 12: 36, 14: 36,
    }),
    extension: new Uint32Array(0),
    fineTransport: new Uint32Array(0),
    fineTopology: new Uint32Array(0),
    fineRedistance: new Uint32Array(0),
    fineVolume: new Uint32Array(0),
    fluidResidency: new Uint32Array(0),
    fluidBulkResidency: new Uint32Array(0),
    adaptive: {
      candidateAuthority,
      acceptedGraph,
      candidateGraph: words(32, { 4: 146 }),
      phiControl: words(20, {
        0: 0x4150_4849, 1: 17, 2: 36, 3: 36, 4: 2764, 5: 1886, 7: 1,
      }),
      phiReceipts,
      velocityReceipts,
      renderer: words(8, { 0: 0x8000_0000, 1: 36, 2: 1886 }),
      massControl: words(32, { 0: 0x414d_4153, 12: 1 }),
      massReceipts: words(32, { 12: 1, 17: 0xffff_ffff }),
      velocityMigration: words(8, { 0: 5, 1: 4306, 4: 8, 6: 8 }),
    },
  };
}

test("epoch-zero candidate authority makes stale candidate receipts dormant", () => {
  const record = adaptiveRecord(0);
  assert.deepEqual(losassoStepSnapshotFailures(record), []);
  const summary = losassoStepSnapshotDiagnosticSummary(record);
  assert.match(summary, /candidate=none/);
  assert.match(summary, /candidateTransaction=none/);
  assert.doesNotMatch(summary, /fatal adaptive candidate transaction/);
});

test("a nonzero candidate epoch keeps the hard fail-closed verdict", () => {
  const failures = losassoStepSnapshotFailures(adaptiveRecord(18));
  assert.ok(failures.some((failure) => failure.startsWith("fatal adaptive candidate transaction:")));
});
