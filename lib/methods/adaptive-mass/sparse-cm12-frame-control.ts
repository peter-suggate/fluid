/** GPU-owned B16/P16 frame authority, version FCA1. */
export const SPARSE_CM12_FRAME_CONTROL_MAGIC = 0x4643_4131;
export const SPARSE_CM12_FRAME_CONTROL_VERSION = 1;
export const SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS = 64;
export const SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS = 3;
export const SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT = 14;
export const SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS =
  SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS
  + SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT;
export const SPARSE_CM12_FRAME_CONTROL_TOTAL_BYTES = 4 * SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS;
export const SPARSE_CM12_FRAME_CONTROL_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FRAME_CONTROL_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
  d4Capable: 1 << 2,
  rigidCapable: 1 << 3,
  boundaryCapable: 1 << 4,
} as const);

export const SPARSE_CM12_FRAME_CONTROL_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  sealed: 3,
  fault: 4,
} as const);

export const SPARSE_CM12_FRAME_CONTROL_COVERAGE = Object.freeze({
  body: 1 << 0,
  boundary: 1 << 1,
  scalarD4: 1 << 2,
  faceD4: 1 << 3,
  scalarOutput: 1 << 4,
  faceOutput: 1 << 5,
  authority: (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
  output: (1 << 4) | (1 << 5),
} as const);

export const SPARSE_CM12_FRAME_CONTROL_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  invalidPhase: 2,
  generationExhausted: 3,
  staleGeneration: 4,
  capability: 5,
  bodyCapacity: 6,
  missingEvidence: 7,
  incompleteOutput: 8,
  velocityExtensionReceipt: 9,
} as const);

export const SPARSE_CM12_FRAME_CONTROL_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  totalWords: 3,
  flags: 4,
  brickFineResolution: 5,
  presentationPageResolution: 6,
  indirectWords: 7,
  familyCount: 8,
  indirectBase: 9,
  cellWorkgroups: 10,
  rowWorkgroups: 11,
  bodyCapacity: 12,
  phase: 13,
  acceptedGeneration: 14,
  candidateGeneration: 15,
  scalarParity: 16,
  faceParity: 17,
  bodyGeneration: 18,
  bodyCount: 19,
  boundaryGeneration: 20,
  boundaryLive: 21,
  d4Generation: 22,
  scalarD4Authority: 23,
  faceD4Authority: 24,
  d4InvalidationCause: 25,
  d4InvalidationOwner: 26,
  coverage: 27,
  requiredAuthorityCoverage: 28,
  requiredOutputCoverage: 29,
  fault: 30,
  firstFaultOwner: 31,
  sealedGeneration: 32,
  committedFrames: 33,
  reservedBase: 34,
} as const);

export const SPARSE_CM12_FRAME_CONTROL_FAMILY = Object.freeze({
  scalarD4Work: 0,
  scalarD4Bypass: 1,
  faceD4Work: 2,
  faceD4Bypass: 3,
  solidCellWork: 4,
  solidCellBypass: 5,
  solidRowWork: 6,
  solidRowBypass: 7,
  bodyWork: 8,
  bodyBypass: 9,
  frameWork: 10,
  frameBlocked: 11,
  bodyRowWork: 12,
  bodyRowBypass: 13,
} as const);

export type SparseCM12FrameControlFamilyName = keyof typeof SPARSE_CM12_FRAME_CONTROL_FAMILY;

export interface SparseCM12FrameControlLayout {
  readonly baseWords: number;
  readonly brickFineResolution: 16;
  readonly presentationPageResolution: 16;
  readonly d4Capable: boolean;
  readonly rigidCapable: boolean;
  readonly boundaryCapable: boolean;
  readonly cellWorkgroups: number;
  readonly rowWorkgroups: number;
  readonly bodyCapacity: number;
  readonly initialGeneration: number;
  readonly indirectBaseWords: number;
  readonly controlWords: number;
  readonly controlBytes: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12FrameControl {
  readonly layout: SparseCM12FrameControlLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12FrameControlByteMapEntry {
  readonly name: string;
  readonly offsetBytes: number;
  readonly sizeBytes: number;
  readonly usage: "atomic-control" | "indirect-triplet";
}

const checkedU32 = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0) || value > 0xffff_fffe) {
    throw new RangeError(`${label} must be ${positive ? "a positive" : "a non-negative"} u32`);
  }
  return value;
};

export const sparseCM12FrameControlIndirectWord = (
  family: SparseCM12FrameControlFamilyName | number,
): number => {
  const index = typeof family === "number" ? family : SPARSE_CM12_FRAME_CONTROL_FAMILY[family];
  if (!Number.isInteger(index) || index < 0 || index >= SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT) {
    throw new RangeError(`invalid frame-control family ${family}`);
  }
  return SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS
    + SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS * index;
};

export function createSparseCM12FrameControl(
  options: {
    readonly brickFineResolution?: 16;
    readonly presentationPageResolution?: 16;
    readonly d4Capable?: boolean;
    readonly rigidCapable?: boolean;
    readonly boundaryCapable?: boolean;
    readonly cellWorkgroups: number;
    readonly rowWorkgroups: number;
    readonly bodyCapacity?: number;
    readonly initialGeneration?: number;
    readonly initialScalarD4Authority?: boolean;
    readonly initialFaceD4Authority?: boolean;
    /** Absolute placement in a larger u32 storage arena. */
    readonly baseWords?: number;
  },
): SparseCM12FrameControl {
  const brickFineResolution = options.brickFineResolution ?? 16;
  const presentationPageResolution = options.presentationPageResolution ?? 16;
  if (brickFineResolution !== 16 || presentationPageResolution !== 16) {
    throw new Error("FCA1 is intentionally the B16/P16 physical ABI");
  }
  const cellWorkgroups = checkedU32(options.cellWorkgroups, "cellWorkgroups");
  const rowWorkgroups = checkedU32(options.rowWorkgroups, "rowWorkgroups");
  const bodyCapacity = checkedU32(options.bodyCapacity ?? 0, "bodyCapacity");
  const initialGeneration = checkedU32(options.initialGeneration ?? 1,
    "initialGeneration", true);
  const baseWords = checkedU32(options.baseWords ?? 0, "baseWords");
  if (initialGeneration >= 0x7fff_fffe) throw new RangeError("initial generation is exhausted");
  const d4Capable = options.d4Capable ?? false;
  const rigidCapable = options.rigidCapable ?? false;
  const boundaryCapable = options.boundaryCapable ?? false;
  const initialScalarD4Authority = options.initialScalarD4Authority ?? false;
  const initialFaceD4Authority = options.initialFaceD4Authority ?? false;
  if ((initialScalarD4Authority || initialFaceD4Authority) && !d4Capable) {
    throw new Error("initial D4 authority requires the static D4 capability");
  }
  const flags = SPARSE_CM12_FRAME_CONTROL_FLAG.complete
    | SPARSE_CM12_FRAME_CONTROL_FLAG.validated
    | (d4Capable ? SPARSE_CM12_FRAME_CONTROL_FLAG.d4Capable : 0)
    | (rigidCapable ? SPARSE_CM12_FRAME_CONTROL_FLAG.rigidCapable : 0)
    | (boundaryCapable ? SPARSE_CM12_FRAME_CONTROL_FLAG.boundaryCapable : 0);
  const layout: SparseCM12FrameControlLayout = Object.freeze({
    baseWords,
    brickFineResolution, presentationPageResolution, d4Capable, rigidCapable,
    boundaryCapable, cellWorkgroups, rowWorkgroups, bodyCapacity, initialGeneration,
    indirectBaseWords: baseWords + SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS,
    controlWords: SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS,
    controlBytes: SPARSE_CM12_FRAME_CONTROL_TOTAL_BYTES,
    totalWords: baseWords + SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS,
    totalBytes: 4 * (baseWords + SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS),
  });
  const words = new Uint32Array(layout.totalWords);
  words.set([
    SPARSE_CM12_FRAME_CONTROL_MAGIC, SPARSE_CM12_FRAME_CONTROL_VERSION,
    SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS, layout.controlWords, flags,
    brickFineResolution, presentationPageResolution,
    SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS, SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
    SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS, cellWorkgroups, rowWorkgroups, bodyCapacity,
    SPARSE_CM12_FRAME_CONTROL_PHASE.accepted,
    initialGeneration, initialGeneration, 0, 0,
    initialGeneration, 0, initialGeneration, 0,
    initialGeneration, initialScalarD4Authority ? 1 : 0,
    initialFaceD4Authority ? 1 : 0, 0, SPARSE_CM12_FRAME_CONTROL_INVALID,
    SPARSE_CM12_FRAME_CONTROL_COVERAGE.authority
      | SPARSE_CM12_FRAME_CONTROL_COVERAGE.output,
    SPARSE_CM12_FRAME_CONTROL_COVERAGE.authority,
    SPARSE_CM12_FRAME_CONTROL_COVERAGE.output,
    SPARSE_CM12_FRAME_CONTROL_FAULT.none, SPARSE_CM12_FRAME_CONTROL_INVALID,
    initialGeneration, 0,
  ], baseWords);
  for (let family = 0; family < SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT; family += 1) {
    const at = baseWords + sparseCM12FrameControlIndirectWord(family);
    words[at] = 0; words[at + 1] = 1; words[at + 2] = 1;
  }
  const control = { layout, words } as const;
  validateSparseCM12FrameControl(control);
  return control;
}

export function sparseCM12FrameControlHeaderValid(control: SparseCM12FrameControl): boolean {
  const { layout: l, words } = control;
  const base = l.baseWords;
  const at = (word: number) => base + word;
  const flags = words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.flags)] ?? 0;
  const capabilityFlags = (l.d4Capable ? SPARSE_CM12_FRAME_CONTROL_FLAG.d4Capable : 0)
    | (l.rigidCapable ? SPARSE_CM12_FRAME_CONTROL_FLAG.rigidCapable : 0)
    | (l.boundaryCapable ? SPARSE_CM12_FRAME_CONTROL_FLAG.boundaryCapable : 0);
  return words.length >= l.totalWords && l.controlWords === SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS
    && l.totalWords === l.baseWords + l.controlWords
    && l.totalBytes === 4 * l.totalWords && l.brickFineResolution === 16
    && l.presentationPageResolution === 16
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.magic)] === SPARSE_CM12_FRAME_CONTROL_MAGIC
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.version)] === SPARSE_CM12_FRAME_CONTROL_VERSION
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.headerWords)]
      === SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.totalWords)] === l.controlWords
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.brickFineResolution)] === 16
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.presentationPageResolution)] === 16
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.indirectWords)]
      === SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.familyCount)]
      === SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.indirectBase)]
      === SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.cellWorkgroups)] === l.cellWorkgroups
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.rowWorkgroups)] === l.rowWorkgroups
    && words[at(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCapacity)] === l.bodyCapacity
    && (flags & (SPARSE_CM12_FRAME_CONTROL_FLAG.complete
      | SPARSE_CM12_FRAME_CONTROL_FLAG.validated))
      === (SPARSE_CM12_FRAME_CONTROL_FLAG.complete
        | SPARSE_CM12_FRAME_CONTROL_FLAG.validated)
    && (flags & (SPARSE_CM12_FRAME_CONTROL_FLAG.d4Capable
      | SPARSE_CM12_FRAME_CONTROL_FLAG.rigidCapable
      | SPARSE_CM12_FRAME_CONTROL_FLAG.boundaryCapable)) === capabilityFlags
    && words.subarray(at(SPARSE_CM12_FRAME_CONTROL_HEADER.reservedBase),
      at(SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS)).every((value) => value === 0);
}

export function validateSparseCM12FrameControl(control: SparseCM12FrameControl): void {
  if (!sparseCM12FrameControlHeaderValid(control)) throw new Error("invalid FCA1 header");
  const { words } = control;
  const base = control.layout.baseWords;
  const phase = words[base + SPARSE_CM12_FRAME_CONTROL_HEADER.phase]!;
  if (phase > SPARSE_CM12_FRAME_CONTROL_PHASE.fault) throw new Error("invalid FCA1 phase");
  for (const name of ["scalarParity", "faceParity", "boundaryLive",
    "scalarD4Authority", "faceD4Authority"] as const) {
    if (words[base + SPARSE_CM12_FRAME_CONTROL_HEADER[name]]! > 1) {
      throw new Error(`invalid FCA1 ${name}`);
    }
  }
  for (let family = 0; family < SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT; family += 1) {
    const at = base + sparseCM12FrameControlIndirectWord(family);
    if (words[at + 1] !== 1 || words[at + 2] !== 1) {
      throw new Error(`invalid FCA1 indirect triplet ${family}`);
    }
  }
}

export function sparseCM12FrameControlByteMap(
  control: SparseCM12FrameControl,
): readonly SparseCM12FrameControlByteMapEntry[] {
  const result: SparseCM12FrameControlByteMapEntry[] = [{
    name: "FCA1 atomic authority header", offsetBytes: 4 * control.layout.baseWords,
    sizeBytes: 4 * SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS, usage: "atomic-control",
  }];
  for (const [name, family] of Object.entries(SPARSE_CM12_FRAME_CONTROL_FAMILY)) {
    result.push({ name, offsetBytes: 4 * (control.layout.baseWords
      + sparseCM12FrameControlIndirectWord(family)),
      sizeBytes: 12, usage: "indirect-triplet" });
  }
  if (result[result.length - 1]!.offsetBytes + 12 !== control.layout.totalBytes) {
    throw new Error("FCA1 byte map does not cover the arena");
  }
  return Object.freeze(result.map((entry) => Object.freeze(entry)));
}

export function corruptSparseCM12FrameControlWord(
  control: SparseCM12FrameControl,
  word: number,
  value: number,
): SparseCM12FrameControl {
  const words = Uint32Array.from(control.words);
  words[control.layout.baseWords + word] = value >>> 0;
  return { layout: control.layout, words };
}
