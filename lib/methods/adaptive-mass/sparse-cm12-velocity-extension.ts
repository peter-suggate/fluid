/** GPU-resident local eight-edge velocity-extension recurrence. */
export interface SparseCM12VelocityExtensionLayout {
  readonly headerBaseWords: number;
  readonly rootStampBaseWords: number;
  readonly blastStampBaseWords: number;
  readonly blastDepthBaseWords: number;
  readonly candidateDepthBaseWords: number;
  readonly rootCauseBaseWords: number;
  readonly rootListBaseWords: number;
  readonly frontierABaseWords: number;
  readonly frontierBBaseWords: number;
  readonly blastListBaseWords: number;
  readonly acceptedDepthBaseWords: number;
  readonly acceptedOwnerBaseWords: number;
  readonly reuseStampBaseWords: number;
  readonly cellCapacity: number;
  readonly totalWords: number;
}

export interface SparseCM12VelocityExtensionStateLayout {
  readonly acceptedVelocityFloatBase: number;
  readonly floatCount: number;
}

export const SPARSE_CM12_VELOCITY_EXTENSION_MAGIC = 0x56455831; // VEX1
export const SPARSE_CM12_VELOCITY_EXTENSION_VERSION = 1;
export const SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS = 32;
export const SPARSE_CM12_VELOCITY_EXTENSION_DEPTH = 8;

export const SPARSE_CM12_VELOCITY_EXTENSION_FLAG = Object.freeze({
  /** Root producers may append work for `candidateGeneration`. */
  collectingRoots: 1 << 0,
  /** Root collection is sealed and the compact recurrence may execute. */
  recurrenceSealed: 1 << 1,
  /** Provenance/capacity/generation failed; no candidate may publish. */
  fault: 1 << 2,
} as const);

export const SPARSE_CM12_VELOCITY_EXTENSION_CAUSE = Object.freeze({
  bootstrap: 1 << 0,
  projectedVelocity: 1 << 1,
  liquidClassification: 1 << 2,
  topologyOrEdgeOwnership: 1 << 3,
  injection: 1 << 4,
  movingSolid: 1 << 5,
} as const);

/** Every mutation that can change a legacy eight-sweep input must author one
 * of these roots before the candidate blast is sealed. */
export const SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT = Object.freeze([
  Object.freeze({ producer: "construction bootstrap", cause: "bootstrap",
    coverage: "every construction-active accepted cell; dormant receivers carry no work" }),
  Object.freeze({ producer: "final projected/collocated velocity",
    cause: "projectedVelocity",
    coverage: "bit-changed liquid cell incident to accepted air" }),
  Object.freeze({ producer: "liquid classification or density authority",
    cause: "liquidClassification", coverage: "changed cell and incident endpoints" }),
  Object.freeze({ producer: "topology or extrapolation-edge ownership",
    cause: "topologyOrEdgeOwnership",
    coverage: "entering active cell and every active endpoint incident to entering/retired ownership; retired cache identity is invalidated locally" }),
  Object.freeze({ producer: "liquid injection", cause: "injection",
    coverage: "every written cell and incident endpoints" }),
  Object.freeze({ producer: "moving-solid velocity/open-fraction update",
    cause: "movingSolid", coverage: "affected cell and both row endpoints" }),
] as const);

export type SparseCM12VelocityExtensionBank = "source" | "destination";

/** Legacy ping-pong bank read by one of the eight extrapolation sweeps. */
export function sparseCM12VelocityExtensionInputBank(
  depth: number,
): SparseCM12VelocityExtensionBank {
  if (!Number.isInteger(depth) || depth < 1
    || depth > SPARSE_CM12_VELOCITY_EXTENSION_DEPTH) {
    throw new RangeError("velocity-extension depth must be in [1, 8]");
  }
  return (depth & 1) === 1 ? "destination" : "source";
}

/** Legacy ping-pong bank authored by one of the eight extrapolation sweeps. */
export function sparseCM12VelocityExtensionOutputBank(
  depth: number,
): SparseCM12VelocityExtensionBank {
  return sparseCM12VelocityExtensionInputBank(depth) === "destination"
    ? "source" : "destination";
}

export const SPARSE_CM12_VELOCITY_EXTENSION_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  flags: 3,
  acceptedGeneration: 4,
  candidateGeneration: 5,
  rootCount: 6,
  rootDispatchX: 7,
  rootDispatchY: 8,
  rootDispatchZ: 9,
  frontierACount: 10,
  frontierADispatchX: 11,
  frontierADispatchY: 12,
  frontierADispatchZ: 13,
  frontierBCount: 14,
  frontierBDispatchX: 15,
  frontierBDispatchY: 16,
  frontierBDispatchZ: 17,
  blastCount: 18,
  blastDispatchX: 19,
  blastDispatchY: 20,
  blastDispatchZ: 21,
  faultCount: 22,
  firstFaultCell: 23,
  firstFaultDepth: 24,
  maximumDepth: 25,
  executedCellCount: 26,
  reusedCellCount: 27,
  uncoveredWriteCount: 28,
  topologyGeneration: 29,
  capacity: 30,
  reserved: 31,
} as const);

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

/** Dedicated persistent state tail. Four floats retain xyz plus the legacy
 * validity lane; it must never alias the transient pressure/candidate arena. */
export function createSparseCM12VelocityExtensionStateLayout(options: {
  readonly baseFloats: number;
  readonly cellCapacity: number;
}): SparseCM12VelocityExtensionStateLayout {
  const base = integer(options.baseFloats, "baseFloats");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const acceptedVelocityFloatBase = Math.ceil(base / 4) * 4;
  return Object.freeze({
    acceptedVelocityFloatBase,
    floatCount: acceptedVelocityFloatBase + 4 * cellCapacity,
  });
}

export function createSparseCM12VelocityExtensionLayout(options: {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly alignmentWords?: number;
}): SparseCM12VelocityExtensionLayout {
  const alignment = integer(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const base = integer(options.baseWords ?? 0, "baseWords");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const headerBaseWords = Math.ceil(base / alignment) * alignment;
  let cursor = headerBaseWords + SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
  const takeCells = () => {
    const result = cursor;
    cursor += cellCapacity;
    return result;
  };
  const rootStampBaseWords = takeCells();
  const blastStampBaseWords = takeCells();
  const blastDepthBaseWords = takeCells();
  const candidateDepthBaseWords = takeCells();
  const rootCauseBaseWords = takeCells();
  const rootListBaseWords = takeCells();
  const frontierABaseWords = takeCells();
  const frontierBBaseWords = takeCells();
  const blastListBaseWords = takeCells();
  const acceptedDepthBaseWords = takeCells();
  const acceptedOwnerBaseWords = takeCells();
  const reuseStampBaseWords = takeCells();
  return Object.freeze({
    headerBaseWords,
    rootStampBaseWords,
    blastStampBaseWords,
    blastDepthBaseWords,
    candidateDepthBaseWords,
    rootCauseBaseWords,
    rootListBaseWords,
    frontierABaseWords,
    frontierBBaseWords,
    blastListBaseWords,
    acceptedDepthBaseWords,
    acceptedOwnerBaseWords,
    reuseStampBaseWords,
    cellCapacity,
    totalWords: cursor,
  });
}

export function createSparseCM12VelocityExtensionInitialWords(
  layout: SparseCM12VelocityExtensionLayout,
  options: Readonly<{ initialGeneration?: number }> = {},
): Uint32Array {
  const initialGeneration = integer(options.initialGeneration ?? 1, "initialGeneration");
  if (initialGeneration === 0 || initialGeneration >= 0x7fff_fffe) {
    throw new RangeError("initialGeneration must be in [1, 0x7ffffffd]");
  }
  const result = new Uint32Array(layout.totalWords - layout.headerBaseWords);
  const h = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
  result[h.magic] = SPARSE_CM12_VELOCITY_EXTENSION_MAGIC;
  result[h.version] = SPARSE_CM12_VELOCITY_EXTENSION_VERSION;
  result[h.headerWords] = SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
  // Construction owns the only full accepted-cell root publication. Open its
  // generation before encodeInitialPresentation so the ordinary frame path
  // never needs a host bootstrap flag or a global runtime dispatch.
  result[h.flags] = SPARSE_CM12_VELOCITY_EXTENSION_FLAG.collectingRoots;
  result[h.candidateGeneration] = initialGeneration;
  result[h.rootDispatchY] = result[h.rootDispatchZ] = 1;
  result[h.frontierADispatchY] = result[h.frontierADispatchZ] = 1;
  result[h.frontierBDispatchY] = result[h.frontierBDispatchZ] = 1;
  result[h.blastDispatchY] = result[h.blastDispatchZ] = 1;
  result[h.firstFaultCell] = 0xffff_ffff;
  result[h.firstFaultDepth] = 0xffff_ffff;
  result[h.capacity] = layout.cellCapacity;
  result.fill(0xffff_ffff,
    layout.acceptedDepthBaseWords - layout.headerBaseWords,
    layout.acceptedOwnerBaseWords - layout.headerBaseWords + layout.cellCapacity);
  return result;
}

/** Conflict-free placement after the accepted ACT1/PCM/FPL/FPP activity tail
 * and after the optional pressure-journal state tail. FCA1 deliberately lives
 * in topologyArena and is therefore not part of either base calculation. */
export function createSparseCM12VelocityExtensionResidentLayouts(options: {
  readonly activityTailWords: number;
  readonly stateTailFloats: number;
  readonly cellCapacity: number;
}): Readonly<{
  activity: SparseCM12VelocityExtensionLayout;
  state: SparseCM12VelocityExtensionStateLayout;
}> {
  return Object.freeze({
    activity: createSparseCM12VelocityExtensionLayout({
      baseWords: options.activityTailWords,
      cellCapacity: options.cellCapacity,
    }),
    state: createSparseCM12VelocityExtensionStateLayout({
      baseFloats: options.stateTailFloats,
      cellCapacity: options.cellCapacity,
    }),
  });
}
