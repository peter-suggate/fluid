/**
 * Method-neutral ABI for a GPU pressure-solve film.
 *
 * A method owns how it fills and decodes the journal. Core owns the stable
 * record and snapshot layout consumed by renderers, workers, and UI. Keeping
 * this contract here means those consumers never import a solver resident.
 */

export const PRESSURE_JOURNAL_VERSION = 1 as const;
export const PRESSURE_JOURNAL_HEADER_FLOATS = 8;
export const PRESSURE_JOURNAL_ITERATION_FLOATS = 16;

export const PRESSURE_JOURNAL_FIELDS = Object.freeze([
  "pressure", "residual", "preconditioned", "direction",
] as const);

export type PressureJournalField = (typeof PRESSURE_JOURNAL_FIELDS)[number];
export const PRESSURE_JOURNAL_FIELD_COUNT = PRESSURE_JOURNAL_FIELDS.length;

/** Header word meanings mirrored by a method's journal writer. */
export const PRESSURE_JOURNAL_HEADER = Object.freeze({
  iterationCursor: 0,
  snapshotCursor: 1,
  armed: 2,
  version: 3,
} as const);

/** Iteration-record word meanings mirrored by a method's journal writer. */
export const PRESSURE_JOURNAL_RECORD = Object.freeze({
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

export interface PressureJournalCapacity {
  readonly iterationCapacity: number;
  readonly snapshotCapacity: number;
  readonly cellStride: number;
}

export interface PressureJournalLayout extends PressureJournalCapacity {
  readonly iterationsOffset: number;
  readonly snapshotsOffset: number;
  readonly floatCount: number;
}

const EMPTY_LAYOUT: PressureJournalLayout = Object.freeze({
  iterationCapacity: 0,
  snapshotCapacity: 0,
  cellStride: 0,
  iterationsOffset: 0,
  snapshotsOffset: 0,
  floatCount: 0,
});

/** Reserve one journal region; any zero capacity collapses it to no storage. */
export function pressureJournalLayout(
  capacity: Partial<PressureJournalCapacity>,
): PressureJournalLayout {
  const iterationCapacity = Math.max(0, Math.floor(capacity.iterationCapacity ?? 0));
  const snapshotCapacity = Math.max(0, Math.floor(capacity.snapshotCapacity ?? 0));
  const cellStride = Math.max(0, Math.floor(capacity.cellStride ?? 0));
  if (iterationCapacity === 0 || cellStride === 0) return EMPTY_LAYOUT;
  const iterationsOffset = PRESSURE_JOURNAL_HEADER_FLOATS;
  const snapshotsOffset = iterationsOffset
    + iterationCapacity * PRESSURE_JOURNAL_ITERATION_FLOATS;
  const snapshotFloats = snapshotCapacity * PRESSURE_JOURNAL_FIELD_COUNT * cellStride;
  return {
    iterationCapacity,
    snapshotCapacity,
    cellStride,
    iterationsOffset,
    snapshotsOffset,
    floatCount: snapshotsOffset + snapshotFloats,
  };
}

/** Float offset of one snapshot field relative to the journal base. */
export function pressureJournalSnapshotOffset(
  layout: PressureJournalLayout,
  snapshot: number,
  field: number,
): number {
  return layout.snapshotsOffset
    + (snapshot * PRESSURE_JOURNAL_FIELD_COUNT + field) * layout.cellStride;
}

/** Log-spaced snapshot schedule shared by pressure-film implementations. */
export function pressureJournalSchedule(
  iterations: number,
  snapshotCapacity: number,
): readonly number[] {
  if (iterations <= 0 || snapshotCapacity <= 0) return [];
  const wanted = new Set<number>([0, iterations]);
  for (let step = 1; step < iterations; step *= 2) wanted.add(step);
  const ordered = [...wanted].filter((value) => value <= iterations)
    .sort((a, b) => a - b);
  if (ordered.length <= snapshotCapacity) return Object.freeze(ordered);
  return Object.freeze([
    ordered[0]!,
    ...ordered.slice(ordered.length - (snapshotCapacity - 1)),
  ]);
}

export interface PressureJournalIteration {
  readonly iteration: number;
  readonly active: boolean;
  readonly gateOpen: boolean;
  readonly gamma: number;
  readonly alpha: number;
  readonly beta: number;
  readonly recursiveRelativeL2: number;
  readonly guardedRelativeL2: number | undefined;
  readonly executed: number;
  readonly curvatureBreakdown: boolean;
  readonly curvatureCollapses: number;
  readonly snapshot: number | undefined;
}

/** Serializable pressure-film receipt crossing the renderer worker boundary. */
export interface PressureJournal {
  readonly version: typeof PRESSURE_JOURNAL_VERSION;
  readonly armed: boolean;
  readonly encodedIterations: number;
  readonly executedIterations: number;
  readonly firstCrossingIteration: number | undefined;
  readonly initialRelativeL2: number;
  readonly records: readonly PressureJournalIteration[];
  readonly snapshotIterations: readonly number[];
}

/**
 * Method-declared pressure-film behavior used before any solver or device
 * exists. The UI asks the selected method for its authored scrub stops rather
 * than importing its numerical implementation.
 */
export interface PressureJournalDescriptor<Values = Readonly<Record<string, unknown>>> {
  isReserved(values: Values): boolean;
  schedule(values: Values): readonly number[];
  /** Optional parameter mutation the generic UI can offer to reserve storage. */
  readonly reserve?: Readonly<{
    parameter: string;
    value: string | number | boolean;
  }>;
}
