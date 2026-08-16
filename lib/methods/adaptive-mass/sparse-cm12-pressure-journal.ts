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

export const SPARSE_CM12_PRESSURE_JOURNAL_VERSION = 1 as const;

/** Floats of journal header, before the first iteration record. */
export const SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS = 8;

/** Floats per encoded-iteration record. */
export const SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS = 16;

/**
 * Cell fields a snapshot carries, in the order the GPU writes them.
 *
 * `pressure` and `residual` are the two the picture is mostly made of; `z` and
 * `direction` are what make the *mechanism* legible, because the difference
 * between the preconditioned residual and the raw one is precisely what the
 * brick-aggregate hierarchy is doing.
 */
export const SPARSE_CM12_PRESSURE_JOURNAL_FIELDS = Object.freeze([
  "pressure", "residual", "preconditioned", "direction",
] as const);

export type SparseCM12PressureJournalField =
  (typeof SPARSE_CM12_PRESSURE_JOURNAL_FIELDS)[number];

export const SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT =
  SPARSE_CM12_PRESSURE_JOURNAL_FIELDS.length;

/**
 * Header word meanings, mirrored in the WGSL.
 *
 * The two cursors are written by the GPU rather than the host because a
 * dispatch cannot be told which iteration it is: the uniform is written once
 * per frame and WebGPU has no push constant. A cursor the journal kernel
 * increments is therefore the only thing that can index the record, and since
 * that kernel is dispatched exactly once per encoded iteration, the cursor
 * *is* the encoded iteration index.
 */
export const SPARSE_CM12_PRESSURE_JOURNAL_HEADER = Object.freeze({
  iterationCursor: 0,
  snapshotCursor: 1,
  armed: 2,
  version: 3,
} as const);

/** Iteration-record word meanings, mirrored in the WGSL. */
export const SPARSE_CM12_PRESSURE_JOURNAL_RECORD = Object.freeze({
  gateOpen: 0,
  gamma: 1,
  alpha: 2,
  beta: 3,
  residualSquared: 4,
  rhsSquared: 5,
  executed: 6,
  curvatureBreakdown: 7,
  guardedTrueSquared: 8,
  guardedTrueMaximum: 9,
  curvatureCollapses: 10,
  firstCrossing: 11,
  initialTrueSquared: 12,
  initialTrueMaximum: 13,
  recursiveToTrueRatio: 14,
  snapshot: 15,
} as const);

export interface SparseCM12PressureJournalCapacity {
  /**
   * Encoded iterations the journal can record. Sized from the solve's own
   * iteration ceiling plus one, because record 0 is the seed state before any
   * iteration has run.
   */
  readonly iterationCapacity: number;
  /** Whole-field snapshots the journal can hold. */
  readonly snapshotCapacity: number;
  /** Floats between one snapshot field and the next: the template cell count. */
  readonly cellStride: number;
}

export interface SparseCM12PressureJournalLayout
  extends SparseCM12PressureJournalCapacity {
  /** Float offset of the first iteration record, relative to the journal base. */
  readonly iterationsOffset: number;
  /** Float offset of the first snapshot, relative to the journal base. */
  readonly snapshotsOffset: number;
  /** Total floats the journal region occupies. Zero when it is not armed. */
  readonly floatCount: number;
}

const EMPTY_LAYOUT: SparseCM12PressureJournalLayout = Object.freeze({
  iterationCapacity: 0, snapshotCapacity: 0, cellStride: 0,
  iterationsOffset: 0, snapshotsOffset: 0, floatCount: 0,
});

/**
 * Reserve the journal region.
 *
 * A capacity of zero in any dimension collapses the whole region to nothing,
 * so "no journal" is the same code path as "journal", not a branch around it.
 */
export function sparseCM12PressureJournalLayout(
  capacity: Partial<SparseCM12PressureJournalCapacity>,
): SparseCM12PressureJournalLayout {
  const iterationCapacity = Math.max(0, Math.floor(capacity.iterationCapacity ?? 0));
  const snapshotCapacity = Math.max(0, Math.floor(capacity.snapshotCapacity ?? 0));
  const cellStride = Math.max(0, Math.floor(capacity.cellStride ?? 0));
  if (iterationCapacity === 0 || cellStride === 0) return EMPTY_LAYOUT;
  const iterationsOffset = SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS;
  const snapshotsOffset = iterationsOffset
    + iterationCapacity * SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS;
  const snapshotFloats = snapshotCapacity * SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT
    * cellStride;
  return {
    iterationCapacity, snapshotCapacity, cellStride,
    iterationsOffset, snapshotsOffset,
    floatCount: snapshotsOffset + snapshotFloats,
  };
}

/** Float offset of one snapshot field, relative to the journal base. */
export function sparseCM12PressureJournalSnapshotOffset(
  layout: SparseCM12PressureJournalLayout,
  snapshot: number,
  field: number,
): number {
  return layout.snapshotsOffset
    + (snapshot * SPARSE_CM12_PRESSURE_JOURNAL_FIELD_COUNT + field) * layout.cellStride;
}

/**
 * Which encoded iterations get a whole-field snapshot.
 *
 * Powers of two, plus the seed and the final iteration, deduplicated and
 * trimmed to capacity. The seed matters because the first correction is the
 * largest one in the whole solve, and the final one matters because it is the
 * only iteration whose field the projection actually consumed.
 *
 * When capacity is short the *tail* is kept rather than the head: the early
 * doublings are visually near-identical, while the last few carry whatever
 * residual structure survived, which is the part worth looking at.
 */
export function sparseCM12PressureJournalSchedule(
  iterations: number,
  snapshotCapacity: number,
): readonly number[] {
  if (iterations <= 0 || snapshotCapacity <= 0) return [];
  const wanted = new Set<number>([0, iterations]);
  for (let step = 1; step < iterations; step *= 2) wanted.add(step);
  const ordered = [...wanted].filter((value) => value <= iterations)
    .sort((a, b) => a - b);
  if (ordered.length <= snapshotCapacity) return Object.freeze(ordered);
  // Always keep the seed; drop from the dense early doublings inward.
  const kept = [ordered[0]!, ...ordered.slice(ordered.length - (snapshotCapacity - 1))];
  return Object.freeze(kept);
}

export interface SparseCM12PressureJournalIteration {
  /** Encoded iteration index. Record 0 is the seed, before any iteration ran. */
  readonly iteration: number;
  /**
   * Whether this iteration did work.
   *
   * Derived from the executed counter rather than read from the gate: the
   * cadence check that closes the gate runs *inside* the iteration that earned
   * the closure, so the gate as sampled at journal time already reads shut for
   * an iteration that computed a full update. The executed counter only
   * advances on a real update, so its delta is exact.
   */
  readonly active: boolean;
  /** The gate as sampled when the record was written. See `active`. */
  readonly gateOpen: boolean;
  /** r·z, the Krylov curvature numerator. */
  readonly gamma: number;
  readonly alpha: number;
  readonly beta: number;
  /** Recursive ‖r‖ relative to ‖b‖. Not the convergence authority. */
  readonly recursiveRelativeL2: number;
  /**
   * ‖b − Ap‖ relative to ‖b‖ at the last cadence check at or before this
   * iteration, or undefined before the first check. This *is* the authority;
   * the recursive value beside it is what drifts.
   */
  readonly guardedRelativeL2: number | undefined;
  readonly executed: number;
  readonly curvatureBreakdown: boolean;
  readonly curvatureCollapses: number;
  /** Index into the snapshot list, when this iteration carries one. */
  readonly snapshot: number | undefined;
}

export interface SparseCM12PressureJournal {
  readonly version: typeof SPARSE_CM12_PRESSURE_JOURNAL_VERSION;
  readonly armed: boolean;
  /** Encoded iteration ceiling this frame ran under. */
  readonly encodedIterations: number;
  /** Iterations that actually did work before the gate closed. */
  readonly executedIterations: number;
  /** First encoded iteration whose guarded residual crossed tolerance. */
  readonly firstCrossingIteration: number | undefined;
  readonly initialRelativeL2: number;
  readonly records: readonly SparseCM12PressureJournalIteration[];
  /** Encoded iteration index of each snapshot, in snapshot order. */
  readonly snapshotIterations: readonly number[];
}

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
