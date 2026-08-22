/**
 * DCA1: sparse dynamic authority for compiled TRA/VEX closure.
 *
 * Gather appends only nonzero producer packets.  Compilers consume those lists
 * and stamp exact spatial row bundles (`cellPacket*3+axis`) and cell packets.
 * Generation stamps make target clearing proportional to touched packets; no
 * world packet/row scan is part of the protocol.
 */
export const SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC = 0x4443_4131; // DCA1
export const SPARSE_CM12_DYNAMIC_CLOSURE_VERSION = 1;
export const SPARSE_CM12_DYNAMIC_CLOSURE_HEADER_WORDS = 32;
export const SPARSE_CM12_DYNAMIC_CLOSURE_INVALID = 0xffff_ffff;
export const SPARSE_CM12_DYNAMIC_CLOSURE_ROW_PLANES = 6;
export const SPARSE_CM12_DYNAMIC_CLOSURE_CELL_PLANES = 2;

export const SPARSE_CM12_DYNAMIC_CLOSURE_ADDRESSING = Object.freeze({
  packet: "stable TEI packet id: leaf*64+localPacket",
  gather: "publish the staged stable packet id, never the compact gather work rank",
  transform: "IBO/VEX local operators preserve stable packet ids across accepted selectors",
  topologyFlip: "clear prior touched masks before selector flip; inactive stable ids remain allocated but never scheduled",
} as const);

export const SPARSE_CM12_DYNAMIC_CLOSURE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, sourceCapacity: 3, targetCapacity: 4,
  generation: 5, fault: 6, firstFaultPacket: 7,
  surfaceSourceCount: 8, densitySourceCount: 9,
  rowTouchedCount: 10, cellTouchedCount: 11,
  surfaceListBase: 12, densityListBase: 13,
  rowStampBase: 14, rowMaskBase: 15, rowTouchedBase: 16,
  cellStampBase: 17, cellMaskBase: 18, cellTouchedBase: 19,
  indirectBase: 20, totalWords: 21,
} as const);

export const SPARSE_CM12_DYNAMIC_CLOSURE_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, sourceOverflow: 2, rowTargetOverflow: 3,
  cellTargetOverflow: 4, invalidGeneration: 5, targetNotCleared: 6,
} as const);

export const SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT = Object.freeze({
  traCompile: 0, vexCompile: 4, rowScatter: 8, vexSeed: 12, words: 16,
} as const);

export const SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION = Object.freeze({
  intra: "positive lane has local axis > 0: axisBase + composite-grid intra formula",
  equalFace: "positive lane has local axis 0: accepted interface base + tangential rank",
  mixedSeam: "positive owner lane indexes the canonical interface patch's stable accepted row",
  sparseAir: "explicit interface patch owns the one-sided stable accepted row",
  snapshot: "first scatter reads destination density/gamma banks",
  refinement: "second scatter reads refinement input banks after snapshot finalization",
  lifetime: "the same exact row masks persist through both ordered scatters and clear next frame",
  publication: "both resolved-row phases perform gamma scatter directly; no TRA1 per-row stamp decode",
} as const);

export const SPARSE_CM12_DYNAMIC_CLOSURE_DISPATCH_ORDER = Object.freeze([
  "clearSparseCM12DynamicRows + clearSparseCM12DynamicCells (prior target indirects)",
  "beginSparseCM12DynamicClosure",
  "gather publishes nonzero source packets",
  "sealSparseCM12DynamicClosureSources",
  "compileSparseCM12DynamicTRA + compileSparseCM12DynamicVEX (source indirects)",
  "sealSparseCM12DynamicClosureTargets",
  "scatterSparseCM12DynamicGammaSnapshotRows",
  "resident finalizeGammaSnapshot",
  "scatterSparseCM12DynamicGammaRefinementRows",
  "resident finalizeGammaRefinement + seedSparseCM12DynamicVEXFrontier",
] as const);

export interface SparseCM12DynamicClosureLayout {
  readonly baseWords: number;
  readonly sourcePacketCapacity: number;
  readonly targetPacketCapacity: number;
  readonly surfaceListBaseWords: number;
  readonly densityListBaseWords: number;
  readonly rowStampBaseWords: number;
  readonly rowMaskBaseWords: number;
  readonly rowTouchedBaseWords: number;
  readonly cellStampBaseWords: number;
  readonly cellMaskBaseWords: number;
  readonly cellTouchedBaseWords: number;
  readonly indirectBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const checked = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)
    || value >= SPARSE_CM12_DYNAMIC_CLOSURE_INVALID) {
    throw new RangeError(`${label} must be a ${positive ? "positive " : ""}u32`);
  }
  return value;
};
const align64 = (value: number) => Math.ceil(value / 64) * 64;

export function createSparseCM12DynamicClosureLayout(options: Readonly<{
  sourcePacketCapacity: number;
  targetPacketCapacity?: number;
  baseWords?: number;
}>): SparseCM12DynamicClosureLayout {
  const sourcePacketCapacity = checked(options.sourcePacketCapacity,
    "DCA1 sourcePacketCapacity", true);
  const targetPacketCapacity = checked(options.targetPacketCapacity
    ?? sourcePacketCapacity, "DCA1 targetPacketCapacity", true);
  const baseWords = align64(checked(options.baseWords ?? 0, "DCA1 baseWords"));
  const surfaceListBaseWords = align64(baseWords + SPARSE_CM12_DYNAMIC_CLOSURE_HEADER_WORDS);
  const densityListBaseWords = align64(surfaceListBaseWords + sourcePacketCapacity);
  const rowStampBaseWords = align64(densityListBaseWords + sourcePacketCapacity);
  const rowMaskBaseWords = align64(rowStampBaseWords + targetPacketCapacity);
  const rowTouchedBaseWords = align64(rowMaskBaseWords
    + SPARSE_CM12_DYNAMIC_CLOSURE_ROW_PLANES * targetPacketCapacity);
  const cellStampBaseWords = align64(rowTouchedBaseWords + targetPacketCapacity);
  const cellMaskBaseWords = align64(cellStampBaseWords + targetPacketCapacity);
  const cellTouchedBaseWords = align64(cellMaskBaseWords
    + SPARSE_CM12_DYNAMIC_CLOSURE_CELL_PLANES * targetPacketCapacity);
  const indirectBaseWords = align64(cellTouchedBaseWords + targetPacketCapacity);
  const totalWords = align64(indirectBaseWords + SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.words);
  checked(totalWords, "DCA1 totalWords");
  return Object.freeze({ baseWords, sourcePacketCapacity, targetPacketCapacity,
    surfaceListBaseWords, densityListBaseWords, rowStampBaseWords, rowMaskBaseWords,
    rowTouchedBaseWords, cellStampBaseWords, cellMaskBaseWords, cellTouchedBaseWords,
    indirectBaseWords, totalWords, totalBytes: 4 * (totalWords - baseWords) });
}

export function createSparseCM12DynamicClosureInitialWords(
  layout: SparseCM12DynamicClosureLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_DYNAMIC_CLOSURE_HEADER;
  words[h.magic] = SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC;
  words[h.version] = SPARSE_CM12_DYNAMIC_CLOSURE_VERSION;
  words[h.headerWords] = SPARSE_CM12_DYNAMIC_CLOSURE_HEADER_WORDS;
  words[h.sourceCapacity] = layout.sourcePacketCapacity;
  words[h.targetCapacity] = layout.targetPacketCapacity;
  words[h.firstFaultPacket] = SPARSE_CM12_DYNAMIC_CLOSURE_INVALID;
  words[h.surfaceListBase] = layout.surfaceListBaseWords;
  words[h.densityListBase] = layout.densityListBaseWords;
  words[h.rowStampBase] = layout.rowStampBaseWords;
  words[h.rowMaskBase] = layout.rowMaskBaseWords;
  words[h.rowTouchedBase] = layout.rowTouchedBaseWords;
  words[h.cellStampBase] = layout.cellStampBaseWords;
  words[h.cellMaskBase] = layout.cellMaskBaseWords;
  words[h.cellTouchedBase] = layout.cellTouchedBaseWords;
  words[h.indirectBase] = layout.indirectBaseWords;
  words[h.totalWords] = layout.totalWords;
  return words;
}

export interface SparseCM12DynamicClosureMaskTarget {
  readonly packetId: number;
  readonly low: number;
  readonly high: number;
  readonly axis?: 0 | 1 | 2;
}

/** CPU oracle for sparse-list publication, exact OR and touched-list dedup. */
export function compileSparseCM12DynamicClosureReference(options: Readonly<{
  producers: Iterable<Readonly<{ packetId: number; surfaceLow: number; surfaceHigh: number;
    densityLow: number; densityHigh: number }>>;
  tra: (packetId: number, low: number, high: number) =>
    Iterable<SparseCM12DynamicClosureMaskTarget>;
  vex: (packetId: number, low: number, high: number) =>
    Iterable<SparseCM12DynamicClosureMaskTarget>;
}>): Readonly<{
  surfaceSources: readonly number[]; densitySources: readonly number[];
  rowTouched: readonly number[]; cellTouched: readonly number[];
  rowMasks: ReadonlyMap<number, readonly [number, number, number, number, number, number]>;
  cellMasks: ReadonlyMap<number, readonly [number, number]>;
}> {
  const surface: Array<readonly [number, number, number]> = [];
  const density: Array<readonly [number, number, number]> = [];
  for (const producer of options.producers) {
    checked(producer.packetId, "DCA1 producer packet");
    if ((producer.surfaceLow | producer.surfaceHigh) !== 0) {
      surface.push([producer.packetId, producer.surfaceLow >>> 0, producer.surfaceHigh >>> 0]);
    }
    if ((producer.densityLow | producer.densityHigh) !== 0) {
      density.push([producer.packetId, producer.densityLow >>> 0, producer.densityHigh >>> 0]);
    }
  }
  const rows = new Map<number, [number, number, number, number, number, number]>();
  for (const [packet, low, high] of surface) for (const target of options.tra(packet, low, high)) {
    if (target.axis === undefined) throw new Error("DCA1 TRA target requires axis");
    const mask = rows.get(target.packetId) ?? [0, 0, 0, 0, 0, 0];
    mask[2 * target.axis] = (mask[2 * target.axis]! | target.low) >>> 0;
    mask[2 * target.axis + 1] = (mask[2 * target.axis + 1]! | target.high) >>> 0;
    rows.set(target.packetId, mask);
  }
  const cells = new Map<number, [number, number]>();
  for (const [packet, low, high] of density) for (const target of options.vex(packet, low, high)) {
    const mask = cells.get(target.packetId) ?? [0, 0];
    mask[0] = (mask[0] | target.low) >>> 0; mask[1] = (mask[1] | target.high) >>> 0;
    cells.set(target.packetId, mask);
  }
  return Object.freeze({
    surfaceSources: Object.freeze(surface.map(([packet]) => packet)),
    densitySources: Object.freeze(density.map(([packet]) => packet)),
    rowTouched: Object.freeze([...rows.keys()]), cellTouched: Object.freeze([...cells.keys()]),
    rowMasks: rows, cellMasks: cells,
  });
}
