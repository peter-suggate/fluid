/**
 * Per-iteration record of one Sparse CM12 pressure solve — the film the
 * pressure lab scrubs.
 *
 * The receipt beside this file (`sparse-cm12-pressure-receipt.ts`) publishes
 * what the solve *ended up* being: one converged residual, one iteration
 * count, one timing. That is the right shape for a gate and the wrong shape
 * for understanding, because every interesting question about a Krylov solve
 * is about the path rather than the endpoint. Which iteration did the
 * tolerance actually fall on? Did the preconditioner stall, or did curvature
 * collapse and restart? Is the residual that is left sitting on the 2:1 seams
 * or spread through the interior?
 *
 * So this is a journal: one record per *encoded* iteration, plus a small set
 * of whole-field snapshots at log-spaced iterations.
 *
 * Three properties are load-bearing:
 *
 * 1. **Encoded, not executed.** The solve encodes a fixed iteration ceiling
 *    and gates the tail at runtime (`scalars[5]`), so the command buffer
 *    always runs `iterations` times and the last several do nothing. The
 *    journal kernel is deliberately *ungated* and records `active` per record,
 *    which is why the film can show the gate closing instead of drawing a
 *    converged tail as though it had been computed.
 *
 * 2. **Log-spaced snapshots.** Convergence is geometric; linear sampling
 *    spends 90% of its bytes on iterations that look identical. Twelve
 *    snapshots at 0,1,2,4,8,… cover a 128-iteration solve better than any
 *    affordable uniform sampling of it.
 *
 * 3. **Zero cost unarmed.** Every offset below is derived from a capacity
 *    that is zero by default. A solver built without the capability reserves
 *    no floats, and the host encodes no journal dispatch, so an unjournalled
 *    advance is the advance that shipped.
 *
 * The scalars this mirrors are the pipelined Chronopoulos-Gear PCG's own; see
 * `reducePipelinedIteration` in the resident WGSL for who writes each one.
 */

import {
  PRESSURE_JOURNAL_HEADER as SPARSE_CM12_PRESSURE_JOURNAL_HEADER,
  PRESSURE_JOURNAL_ITERATION_FLOATS as SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
  PRESSURE_JOURNAL_RECORD as SPARSE_CM12_PRESSURE_JOURNAL_RECORD,
  PRESSURE_JOURNAL_VERSION as SPARSE_CM12_PRESSURE_JOURNAL_VERSION,
  type PressureJournal as SparseCM12PressureJournal,
  type PressureJournalIteration as SparseCM12PressureJournalIteration,
  type PressureJournalLayout as SparseCM12PressureJournalLayout,
} from "../../core/pressure-journal";

export {
  PRESSURE_JOURNAL_FIELD_COUNT as SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT,
  PRESSURE_JOURNAL_FIELDS as SPARSE_CM12_PRESSURE_JOURNAL_FIELDS,
  PRESSURE_JOURNAL_HEADER as SPARSE_CM12_PRESSURE_JOURNAL_HEADER,
  PRESSURE_JOURNAL_HEADER_FLOATS as SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS,
  PRESSURE_JOURNAL_ITERATION_FLOATS as SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
  PRESSURE_JOURNAL_RECORD as SPARSE_CM12_PRESSURE_JOURNAL_RECORD,
  PRESSURE_JOURNAL_VERSION as SPARSE_CM12_PRESSURE_JOURNAL_VERSION,
  pressureJournalLayout as sparseCM12PressureJournalLayout,
  pressureJournalSchedule as sparseCM12PressureJournalSchedule,
  pressureJournalSnapshotOffset as sparseCM12PressureJournalSnapshotOffset,
  type PressureJournal as SparseCM12PressureJournal,
  type PressureJournalCapacity as SparseCM12PressureJournalCapacity,
  type PressureJournalField as SparseCM12PressureJournalField,
  type PressureJournalIteration as SparseCM12PressureJournalIteration,
  type PressureJournalLayout as SparseCM12PressureJournalLayout,
} from "../../core/pressure-journal";

const relative = (squared: number, rhsSquared: number): number =>
  rhsSquared > 0 ? Math.sqrt(Math.max(0, squared) / rhsSquared) : 0;

/**
 * Decode the journal header and iteration records.
 *
 * Takes only the header-and-records prefix, never the snapshots: those are
 * hundreds of megabytes and the picture reads them on the GPU where they
 * already are. Nothing here needs them.
 */
export function decodeSparseCM12PressureJournal(
  floats: Float32Array,
  layout: SparseCM12PressureJournalLayout,
): SparseCM12PressureJournal {
  const header = SPARSE_CM12_PRESSURE_JOURNAL_HEADER;
  const word = SPARSE_CM12_PRESSURE_JOURNAL_RECORD;
  const stride = SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS;
  const recorded = Math.min(layout.iterationCapacity,
    Math.max(0, Math.round(floats[header.iterationCursor] ?? 0)));
  const armed = (floats[header.armed] ?? 0) > 0.5;
  const records: SparseCM12PressureJournalIteration[] = [];
  const snapshotIterations: number[] = [];
  let executedIterations = 0;
  let firstCrossingIteration: number | undefined;
  let initialRelativeL2 = 0;
  let guardedRelativeL2: number | undefined;
  let previousExecuted = 0;
  for (let index = 0; index < recorded; index += 1) {
    const at = layout.iterationsOffset + index * stride;
    const rhsSquared = floats[at + word.rhsSquared] ?? 0;
    if (index === 0) {
      initialRelativeL2 = relative(floats[at + word.initialTrueSquared] ?? 0, rhsSquared);
    }
    // The guarded value only moves at the cadence; carrying the last one
    // forward is what makes a per-iteration read of it meaningful, and it is
    // labelled `guarded` rather than `true` precisely because of that hold.
    const guardedSquared = floats[at + word.guardedTrueSquared] ?? 0;
    if (guardedSquared > 0) guardedRelativeL2 = relative(guardedSquared, rhsSquared);
    const gateOpen = (floats[at + word.gateOpen] ?? 0) > 0.5;
    const executed = Math.max(0, Math.round(floats[at + word.executed] ?? 0));
    const active = index > 0 && executed > previousExecuted;
    previousExecuted = executed;
    executedIterations = Math.max(executedIterations, executed);
    const crossing = Math.round(floats[at + word.firstCrossing] ?? -1);
    if (crossing >= 0 && firstCrossingIteration === undefined) {
      firstCrossingIteration = crossing;
    }
    const snapshot = Math.round(floats[at + word.snapshot] ?? -1);
    if (snapshot >= 0) {
      snapshotIterations[snapshot] = index;
    }
    records.push({
      iteration: index,
      active,
      gateOpen,
      gamma: floats[at + word.gamma] ?? 0,
      alpha: floats[at + word.alpha] ?? 0,
      beta: floats[at + word.beta] ?? 0,
      recursiveRelativeL2: relative(floats[at + word.residualSquared] ?? 0, rhsSquared),
      guardedRelativeL2,
      executed,
      curvatureBreakdown: (floats[at + word.curvatureBreakdown] ?? 0) > 0.5,
      curvatureCollapses: Math.max(0,
        Math.round(floats[at + word.curvatureCollapses] ?? 0)),
      snapshot: snapshot >= 0 ? snapshot : undefined,
    });
  }
  return {
    version: SPARSE_CM12_PRESSURE_JOURNAL_VERSION,
    armed,
    encodedIterations: Math.max(0, records.length - 1),
    executedIterations,
    firstCrossingIteration,
    initialRelativeL2,
    records: Object.freeze(records),
    snapshotIterations: Object.freeze(
      snapshotIterations.filter((value) => value !== undefined)),
  };
}

/**
 * Where the residual that is left actually lives.
 *
 * The convergence curve says how much residual remains; it cannot say *whose*
 * it is. This bins one snapshot's residual by the class of the cell holding
 * it, which is the question the adaptive method exists to raise: a 2:1 seam
 * couples cells of two different sizes through a ghost-fluid weight, and if
 * that coupling is the part converging slowly then no amount of extra
 * iterations on the interior will help.
 *
 * Host-side on purpose. Binning on the GPU would need atomics in the solve's
 * hot loop for a number twelve snapshots can answer exactly, and this way an
 * unarmed frame pays nothing at all rather than paying a little.
 */
export interface SparseCM12PressureJournalClassNorms {
  readonly counts: Readonly<Record<string, number>>;
  readonly residualL2: Readonly<Record<string, number>>;
  /** Largest |residual| in each class, which a norm hides. */
  readonly residualMaximum: Readonly<Record<string, number>>;
}

export function sparseCM12PressureJournalClassNorms(
  residual: Float32Array,
  classOf: readonly string[] | ((cell: number) => string | undefined),
  cellCount = residual.length,
): SparseCM12PressureJournalClassNorms {
  const counts: Record<string, number> = {};
  const sums: Record<string, number> = {};
  const maxima: Record<string, number> = {};
  const lookup = typeof classOf === "function"
    ? classOf
    : (cell: number) => classOf[cell];
  for (let cell = 0; cell < cellCount; cell += 1) {
    const name = lookup(cell);
    if (name === undefined) continue;
    const value = residual[cell] ?? 0;
    counts[name] = (counts[name] ?? 0) + 1;
    sums[name] = (sums[name] ?? 0) + value * value;
    maxima[name] = Math.max(maxima[name] ?? 0, Math.abs(value));
  }
  const residualL2: Record<string, number> = {};
  for (const [name, sum] of Object.entries(sums)) residualL2[name] = Math.sqrt(sum);
  return {
    counts: Object.freeze(counts),
    residualL2: Object.freeze(residualL2),
    residualMaximum: Object.freeze(maxima),
  };
}

/** Fail fast before a journal reaches a panel or a saved artifact. */
export function assertSparseCM12PressureJournal(
  journal: SparseCM12PressureJournal,
): void {
  if (journal.version !== SPARSE_CM12_PRESSURE_JOURNAL_VERSION) {
    throw new Error(`unsupported Sparse CM12 pressure journal version ${journal.version}`);
  }
  if (journal.records.length === 0) return;
  for (const record of journal.records) {
    if (!Number.isFinite(record.recursiveRelativeL2) || record.recursiveRelativeL2 < 0) {
      throw new Error(`journal iteration ${record.iteration} has an invalid residual`);
    }
  }
  // A gate that reopens would mean the journal recorded the tail out of order,
  // which is the one way the cursor mechanism could silently go wrong.
  let closed = false;
  for (const record of journal.records) {
    if (record.iteration === 0) continue;
    if (!record.active) closed = true;
    else if (closed) {
      throw new Error(
        `journal iteration ${record.iteration} is active after the gate closed`);
    }
  }
}
