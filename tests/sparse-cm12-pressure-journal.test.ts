import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT,
  SPARSE_CM12_PRESSURE_JOURNAL_HEADER,
  SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS,
  SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
  SPARSE_CM12_PRESSURE_JOURNAL_RECORD,
  assertSparseCM12PressureJournal,
  decodeSparseCM12PressureJournal,
  sparseCM12PressureJournalClassNorms,
  sparseCM12PressureJournalLayout,
  sparseCM12PressureJournalSchedule,
  sparseCM12PressureJournalSnapshotOffset,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-journal";
import { webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

const HEADER = SPARSE_CM12_PRESSURE_JOURNAL_HEADER;
const WORD = SPARSE_CM12_PRESSURE_JOURNAL_RECORD;

/** Build a journal buffer the way the GPU kernel would have written it. */
function synthesize(options: {
  readonly iterationCapacity: number;
  readonly snapshotCapacity: number;
  readonly cellStride: number;
  readonly records: readonly {
    readonly gateOpen: boolean;
    readonly executed: number;
    readonly residualSquared: number;
    readonly guardedTrueSquared?: number;
    readonly firstCrossing?: number;
    readonly snapshot?: number;
  }[];
  readonly rhsSquared?: number;
}) {
  const layout = sparseCM12PressureJournalLayout(options);
  const floats = new Float32Array(layout.floatCount);
  floats[HEADER.iterationCursor] = options.records.length;
  floats[HEADER.armed] = 1;
  floats[HEADER.version] = 1;
  options.records.forEach((record, index) => {
    const at = layout.iterationsOffset
      + index * SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS;
    floats[at + WORD.gateOpen] = record.gateOpen ? 1 : 0;
    floats[at + WORD.executed] = record.executed;
    floats[at + WORD.residualSquared] = record.residualSquared;
    floats[at + WORD.rhsSquared] = options.rhsSquared ?? 1;
    floats[at + WORD.guardedTrueSquared] = record.guardedTrueSquared ?? 0;
    floats[at + WORD.firstCrossing] = record.firstCrossing ?? -1;
    floats[at + WORD.snapshot] = record.snapshot ?? -1;
    floats[at + WORD.initialTrueSquared] = index === 0 ? record.residualSquared : 0;
  });
  return { layout, floats };
}

test("a journal reserves nothing until it is asked for", () => {
  const empty = sparseCM12PressureJournalLayout({});
  assert.equal(empty.floatCount, 0);
  assert.equal(sparseCM12PressureJournalLayout(
    { iterationCapacity: 0, snapshotCapacity: 12, cellStride: 1000 }).floatCount, 0);
  // A capacity with nowhere to put a cell is also nothing, rather than a
  // header floating in front of zero-length snapshots.
  assert.equal(sparseCM12PressureJournalLayout(
    { iterationCapacity: 129, snapshotCapacity: 12, cellStride: 0 }).floatCount, 0);
});

test("the reserved region is exactly header, records, then snapshots", () => {
  const layout = sparseCM12PressureJournalLayout(
    { iterationCapacity: 129, snapshotCapacity: 12, cellStride: 1000 });
  assert.equal(layout.iterationsOffset, SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS);
  assert.equal(layout.snapshotsOffset,
    SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS
    + 129 * SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS);
  assert.equal(layout.floatCount, layout.snapshotsOffset
    + 12 * SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT * 1000);
  // Snapshot addressing is field-minor, which is what lets the overlay bind one
  // field of one snapshot as a contiguous run of cell-indexed floats.
  assert.equal(sparseCM12PressureJournalSnapshotOffset(layout, 0, 0),
    layout.snapshotsOffset);
  assert.equal(sparseCM12PressureJournalSnapshotOffset(layout, 0, 1),
    layout.snapshotsOffset + 1000);
  assert.equal(sparseCM12PressureJournalSnapshotOffset(layout, 1, 0),
    layout.snapshotsOffset + SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT * 1000);
});

test("the snapshot schedule is log-spaced and keeps both ends", () => {
  const schedule = sparseCM12PressureJournalSchedule(128, 12);
  assert.deepEqual([...schedule], [0, 1, 2, 4, 8, 16, 32, 64, 128]);
  // Under capacity the seed survives and the *early* doublings are what go:
  // iteration 1 and 2 look alike, the last few do not.
  const tight = sparseCM12PressureJournalSchedule(128, 4);
  assert.equal(tight.length, 4);
  assert.equal(tight[0], 0);
  assert.equal(tight[tight.length - 1], 128);
  assert.deepEqual([...tight], [0, 32, 64, 128]);
  assert.deepEqual([...sparseCM12PressureJournalSchedule(0, 12)], []);
  assert.deepEqual([...sparseCM12PressureJournalSchedule(128, 0)], []);
});

test("an iteration counts as active only when it advanced the executed count", () => {
  // The gate closes inside the iteration that earned the closure, so the gate
  // as sampled reads shut for an iteration that did a full update. Reading
  // activity off the gate would under-report the last real iteration.
  const { layout, floats } = synthesize({
    iterationCapacity: 8, snapshotCapacity: 2, cellStride: 4,
    records: [
      { gateOpen: true, executed: 0, residualSquared: 1 },
      { gateOpen: true, executed: 1, residualSquared: 0.5 },
      // Did work, then its cadence check closed the gate.
      { gateOpen: false, executed: 2, residualSquared: 0.25 },
      { gateOpen: false, executed: 2, residualSquared: 0.25 },
      { gateOpen: false, executed: 2, residualSquared: 0.25 },
    ],
  });
  const journal = decodeSparseCM12PressureJournal(floats, layout);
  assert.deepEqual(journal.records.map((record) => record.active),
    [false, true, true, false, false]);
  assert.deepEqual(journal.records.map((record) => record.gateOpen),
    [true, true, false, false, false]);
  assert.equal(journal.executedIterations, 2);
  assert.equal(journal.encodedIterations, 4);
  assertSparseCM12PressureJournal(journal);
});

test("the guarded residual is held between cadence checks", () => {
  const { layout, floats } = synthesize({
    iterationCapacity: 8, snapshotCapacity: 2, cellStride: 4,
    rhsSquared: 4,
    records: [
      { gateOpen: true, executed: 0, residualSquared: 4 },
      { gateOpen: true, executed: 1, residualSquared: 1 },
      { gateOpen: true, executed: 2, residualSquared: 0.25, guardedTrueSquared: 0.64 },
      { gateOpen: true, executed: 3, residualSquared: 0.0625 },
    ],
  });
  const journal = decodeSparseCM12PressureJournal(floats, layout);
  // Before the first check there is no authority to report.
  assert.equal(journal.records[0]!.guardedRelativeL2, undefined);
  assert.equal(journal.records[1]!.guardedRelativeL2, undefined);
  // Tolerances, not equalities: the journal is an f32 buffer, so every value
  // here has been through a single-precision round trip.
  const near = (value: number | undefined, expected: number) => {
    assert.ok(value !== undefined && Math.abs(value - expected) < 1e-6,
      `${value} is not ${expected}`);
  };
  near(journal.records[2]!.guardedRelativeL2, 0.4);
  // Held, not recomputed: the guard only runs at the cadence.
  near(journal.records[3]!.guardedRelativeL2, 0.4);
  // The recursive value keeps moving beside it, which is exactly the drift the
  // receipt refuses to publish as convergence.
  near(journal.records[3]!.recursiveRelativeL2, 0.125);
});

test("a decode never reads past the cursor", () => {
  const { layout, floats } = synthesize({
    iterationCapacity: 64, snapshotCapacity: 4, cellStride: 4,
    records: [
      { gateOpen: true, executed: 0, residualSquared: 1 },
      { gateOpen: true, executed: 1, residualSquared: 0.5 },
    ],
  });
  const journal = decodeSparseCM12PressureJournal(floats, layout);
  assert.equal(journal.records.length, 2);
  // A capture that never ran leaves an unarmed header, and that must not decode
  // as sixty-four converged iterations.
  const blank = new Float32Array(layout.floatCount);
  const empty = decodeSparseCM12PressureJournal(blank, layout);
  assert.equal(empty.armed, false);
  assert.equal(empty.records.length, 0);
});

test("snapshot stamps resolve to their encoded iterations", () => {
  const { layout, floats } = synthesize({
    iterationCapacity: 8, snapshotCapacity: 3, cellStride: 4,
    records: [
      { gateOpen: true, executed: 0, residualSquared: 1, snapshot: 0 },
      { gateOpen: true, executed: 1, residualSquared: 0.5, snapshot: 1 },
      { gateOpen: true, executed: 2, residualSquared: 0.25 },
      { gateOpen: true, executed: 3, residualSquared: 0.1, snapshot: 2 },
    ],
  });
  const journal = decodeSparseCM12PressureJournal(floats, layout);
  assert.deepEqual([...journal.snapshotIterations], [0, 1, 3]);
  assert.equal(journal.records[2]!.snapshot, undefined);
  assert.equal(journal.records[3]!.snapshot, 2);
});

test("a gate that reopens is rejected rather than drawn", () => {
  const { layout, floats } = synthesize({
    iterationCapacity: 8, snapshotCapacity: 2, cellStride: 4,
    records: [
      { gateOpen: true, executed: 0, residualSquared: 1 },
      { gateOpen: true, executed: 1, residualSquared: 0.5 },
      { gateOpen: false, executed: 1, residualSquared: 0.5 },
      // Executed advancing again after a gated iteration is the one way the
      // cursor mechanism could have recorded the tail out of order.
      { gateOpen: true, executed: 2, residualSquared: 0.25 },
    ],
  });
  const journal = decodeSparseCM12PressureJournal(floats, layout);
  assert.throws(() => assertSparseCM12PressureJournal(journal),
    /active after the gate closed/);
});

test("residual norms bin by the class of the cell holding them", () => {
  const residual = Float32Array.from([0.3, -0.4, 0.1, 2, -1]);
  const classes = ["regular", "regular", "mixed-seam", "mixed-seam", "brick-face"];
  const norms = sparseCM12PressureJournalClassNorms(residual, classes);
  assert.equal(norms.counts["regular"], 2);
  assert.equal(norms.counts["mixed-seam"], 2);
  assert.ok(Math.abs(norms.residualL2["regular"]! - 0.5) < 1e-6);
  assert.ok(Math.abs(norms.residualL2["mixed-seam"]! - Math.sqrt(4.01)) < 1e-6);
  // A norm hides which single cell carries it, so the maximum is reported too.
  assert.equal(norms.residualMaximum["mixed-seam"], 2);
  assert.equal(norms.residualMaximum["brick-face"], 1);
  // A cell with no class is not a zero-residual cell; it is not this method's
  // unknown at all, and must not dilute anyone's norm.
  const partial = sparseCM12PressureJournalClassNorms(residual,
    (cell) => cell < 2 ? "regular" : undefined);
  assert.equal(partial.counts["regular"], 2);
  assert.equal(Object.keys(partial.counts).length, 1);
});

test("the shader and the host agree on the journal layout", () => {
  // The record layout lives in two places by necessity — a WGSL literal and a
  // TypeScript map — and nothing but a test can keep them in step. A drift here
  // would decode a film of plausible, wrong numbers rather than failing.
  const constant = (name: string): number => {
    const match = webgpuSparseCM12ResidentWGSL.match(
      new RegExp(`const ${name}:u32=(\\d+)u;`));
    assert.ok(match, `${name} is not declared in the resident WGSL`);
    return Number(match![1]);
  };
  assert.equal(constant("JOURNAL_HEADER_FLOATS"),
    SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS);
  assert.equal(constant("JOURNAL_ITERATION_FLOATS"),
    SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS);
  assert.equal(constant("JOURNAL_FIELD_COUNT"),
    SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT);

  // Every record word the kernel writes, in the order it writes them.
  const body = webgpuSparseCM12ResidentWGSL.slice(
    webgpuSparseCM12ResidentWGSL.indexOf("fn journalIteration("));
  const writes = [...body.slice(0, body.indexOf("\n}")).matchAll(
    /state\[at(?:\+(\d+)u)?\]=/g)].map((match) => Number(match[1] ?? 0));
  assert.equal(writes.length, SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
    "the kernel must write every record word it reserves");
  assert.deepEqual(writes,
    [...Array(SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS).keys()]);
  // The two the decode depends on most, pinned by name against the WGSL source.
  assert.match(body, new RegExp(`state\\[at\\+${WORD.executed}u\\]=scalars\\[12\\]`));
  assert.match(body, new RegExp(`state\\[at\\+${WORD.snapshot}u\\]=snapshot`));
});
