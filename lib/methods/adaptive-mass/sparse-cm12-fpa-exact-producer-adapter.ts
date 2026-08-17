import { SPARSE_CM12_FACE_PROJECTION_CAUSE } from
  "./sparse-cm12-face-projection-authority";

/** FPE1 is producer ingress for FPA1, never a competing result authority. */
export const SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC = 0x4650_4531; // FPE1
export const SPARSE_CM12_FPA_EXACT_PRODUCER_VERSION = 1;
export const SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER_WORDS = 32;
export const SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS = 8;
export const SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FPA_EXACT_PRODUCER_PHASE = Object.freeze({
  uninitialized: 0, accepted: 1, collecting: 2, covering: 3, sealed: 4, fault: 5,
} as const);

export const SPARSE_CM12_FPA_EXACT_PRODUCER_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, generationGap: 3,
  invalidFamily: 4, invalidId: 5, nonChangeEvent: 6,
  receiptOverflow: 7, coverageGap: 8, uncoveredWrite: 9,
  fpaMarkRejected: 10, reverseDependency: 11,
} as const);

export const SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY = Object.freeze({
  liquidPhaseCell: 0,
  vexCell: 1,
  sourceFaceRow: 2,
  topologyCell: 3,
  movingSolidCell: 4,
  movingSolidRow: 5,
  policyRow: 6,
} as const);
export type SparseCM12FPAExactProducerFamilyName = keyof
  typeof SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY;
export const SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT = 7;

export const SPARSE_CM12_FPA_EXACT_PRODUCER_CAUSE = Object.freeze({
  liquidPhaseCell: SPARSE_CM12_FACE_PROJECTION_CAUSE.densityPhaseBits,
  vexCell: SPARSE_CM12_FACE_PROJECTION_CAUSE.velocityBits
    | SPARSE_CM12_FACE_PROJECTION_CAUSE.preparationDependency,
  sourceFaceRow: SPARSE_CM12_FACE_PROJECTION_CAUSE.sourceFaceBits,
  topologyCell: SPARSE_CM12_FACE_PROJECTION_CAUSE.topology,
  movingSolidCell: SPARSE_CM12_FACE_PROJECTION_CAUSE.movingSolid,
  movingSolidRow: SPARSE_CM12_FACE_PROJECTION_CAUSE.movingSolid,
  policyRow: SPARSE_CM12_FACE_PROJECTION_CAUSE.characteristicPolicy,
} satisfies Readonly<Record<SparseCM12FPAExactProducerFamilyName, number>>);

export const SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, familyWords: 3, familyCount: 4,
  phase: 5, acceptedGeneration: 6, candidateGeneration: 7,
  frameGeneration: 8, fault: 9, firstFaultFamily: 10, firstFaultId: 11,
  expectedReceipts: 12, coveredReceipts: 13, producerEventCount: 14,
  preparationRowWriteCount: 15, pendingEventCount: 16, causeMask: 17,
  familyBaseWords: 18, flags: 19,
} as const);

export const SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_HEADER = Object.freeze({
  generation: 0, expectedReceipts: 1, coveredReceipts: 2,
  producerEventCount: 3, preparationRowWriteCount: 4, causeMask: 5,
  firstProducerId: 6, lastProducerId: 7,
} as const);

export interface SparseCM12FPAExactProducerAdapterLayout {
  readonly baseWords: number;
  readonly familyBaseWords: number;
  readonly cellCapacity: number;
  readonly rowCapacity: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0x3fff_ffff) {
    throw new RangeError(`${label} must be an integer in [0,2^30)`);
  }
  return value;
};
const aligned = (value: number) => Math.ceil(value / 64) * 64;

export function createSparseCM12FPAExactProducerAdapterLayout(options: {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly rowCapacity: number;
}): SparseCM12FPAExactProducerAdapterLayout {
  const baseWords = aligned(checked(options.baseWords ?? 0, "FPE1 baseWords"));
  const cellCapacity = checked(options.cellCapacity, "FPE1 cellCapacity");
  const rowCapacity = checked(options.rowCapacity, "FPE1 rowCapacity");
  if (cellCapacity === 0 || rowCapacity === 0) {
    throw new RangeError("FPE1 cell and row capacities must be positive");
  }
  const familyBaseWords = baseWords + SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER_WORDS;
  const totalWords = aligned(familyBaseWords
    + SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT
      * SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS);
  return Object.freeze({ baseWords, familyBaseWords, cellCapacity, rowCapacity,
    totalWords, totalBytes: 4 * totalWords });
}

export function createSparseCM12FPAExactProducerAdapterInitialWords(
  layout: SparseCM12FPAExactProducerAdapterLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER;
  words[h.magic] = SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC;
  words[h.version] = SPARSE_CM12_FPA_EXACT_PRODUCER_VERSION;
  words[h.headerWords] = SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER_WORDS;
  words[h.familyWords] = SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS;
  words[h.familyCount] = SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT;
  words[h.phase] = SPARSE_CM12_FPA_EXACT_PRODUCER_PHASE.accepted;
  words[h.firstFaultFamily] = SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID;
  words[h.firstFaultId] = SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID;
  words[h.familyBaseWords] = layout.familyBaseWords;
  for (let family = 0; family < SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT;
    family += 1) {
    const at = layout.familyBaseWords - layout.baseWords
      + family * SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS;
    words[at + SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_HEADER.firstProducerId] =
      SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID;
  }
  return words;
}

export type SparseCM12FPAExactProducerEvent = Readonly<{
  family: SparseCM12FPAExactProducerFamilyName;
  stableId: number;
  generation: number;
  beforeBits: readonly number[];
  afterBits: readonly number[];
}>;

export interface SparseCM12FPAExactProducerCPUResult {
  readonly generation: number;
  readonly preparationRows: readonly number[];
  readonly expectedReceipts: Readonly<Record<SparseCM12FPAExactProducerFamilyName, number>>;
  readonly coveredReceipts: Readonly<Record<SparseCM12FPAExactProducerFamilyName, number>>;
  readonly rowWrites: Readonly<Record<SparseCM12FPAExactProducerFamilyName, number>>;
  readonly causeMask: number;
}

const familyNames = Object.keys(SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY) as
  SparseCM12FPAExactProducerFamilyName[];

/** CPU oracle for exact change gating, receipts, and stable row union. */
export function expandSparseCM12FPAExactProducerEvents(options: {
  readonly layout: SparseCM12FPAExactProducerAdapterLayout;
  readonly generation: number;
  readonly events: readonly SparseCM12FPAExactProducerEvent[];
  readonly rowsForEvent: (event: SparseCM12FPAExactProducerEvent) => readonly number[];
  readonly expectedReceipts?: Partial<Readonly<
    Record<SparseCM12FPAExactProducerFamilyName, number>>>;
}): SparseCM12FPAExactProducerCPUResult {
  const generation = checked(options.generation, "FPE1 generation");
  if (generation === 0) throw new RangeError("FPE1 generation must be positive");
  const expected = Object.fromEntries(familyNames.map((name) => [name,
    options.expectedReceipts?.[name]
      ?? options.events.filter((event) => event.family === name).length])) as
    Record<SparseCM12FPAExactProducerFamilyName, number>;
  const covered = Object.fromEntries(familyNames.map((name) => [name, 0])) as
    Record<SparseCM12FPAExactProducerFamilyName, number>;
  const rowWrites = Object.fromEntries(familyNames.map((name) => [name, 0])) as
    Record<SparseCM12FPAExactProducerFamilyName, number>;
  const rows = new Set<number>(); let causeMask = 0;
  for (const event of options.events) {
    if (event.generation !== generation) throw new Error("FPE1 event generation gap");
    const capacity = event.family.endsWith("Row")
      ? options.layout.rowCapacity : options.layout.cellCapacity;
    if (!Number.isSafeInteger(event.stableId) || event.stableId < 0
      || event.stableId >= capacity) throw new RangeError("FPE1 producer id is invalid");
    if (event.beforeBits.length !== event.afterBits.length
      || event.beforeBits.every((bits, index) => bits === event.afterBits[index])) {
      throw new Error("FPE1 producer event does not describe an exact bit change");
    }
    for (const row of options.rowsForEvent(event)) {
      if (!Number.isSafeInteger(row) || row < 0 || row >= options.layout.rowCapacity) {
        throw new RangeError("FPE1 expanded row is invalid");
      }
      rows.add(row); rowWrites[event.family] += 1;
    }
    covered[event.family] += 1;
    if (covered[event.family] > expected[event.family]) {
      throw new Error(`FPE1 ${event.family} receipt overflow`);
    }
    causeMask |= SPARSE_CM12_FPA_EXACT_PRODUCER_CAUSE[event.family];
  }
  for (const family of familyNames) if (covered[family] !== expected[family]) {
    throw new Error(`FPE1 ${family} coverage gap`);
  }
  return Object.freeze({ generation,
    preparationRows: Object.freeze([...rows].sort((a, b) => a - b)),
    expectedReceipts: Object.freeze(expected), coveredReceipts: Object.freeze(covered),
    rowWrites: Object.freeze(rowWrites), causeMask });
}

export const SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST = Object.freeze({
  authority: "FPA1",
  role: "producer-ingress-only",
  producerFamilies: Object.freeze([...familyNames]),
  requiredHooks: Object.freeze([
    "fpeaFrameGeneration", "fpeaExpectedProducerReceipts",
    "fpeaIncidenceBegin", "fpeaIncidenceEnd",
    "fpeaIncidenceRow", "fpeaVexReverseProvenanceValid",
    "fpeaVexReverseBegin", "fpeaVexReverseEnd",
    "fpeaVexReverseRow", "fpaPreparationRowLive", "fpaMarkPreparationRow",
    "fpaCoverPreparationReceipt",
  ]),
  providedEntryPoints: Object.freeze([
    "beginSparseCM12FPAExactProducerAdapter",
    "sealSparseCM12FPAExactProducerAdapter",
    "acceptSparseCM12FPAExactProducerAdapter",
  ]),
  invariants: Object.freeze([
    "no accepted-domain scan", "no fallback dispatch", "no CPU authority",
    "one FPA receipt per exact producer event", "fault leaves adapter unaccepted",
  ]),
});
