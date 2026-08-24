/** FPP1: compact FPL1-driven, page-transactional presentation executor. */
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC = 0x4650_5031; // FPP1
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION = 1;
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS = 32;
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS = 12;
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_STAGE = 5;
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE = 64;

export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, pageWords: 3,
  brickFineResolution: 4, pageResolution: 5, tilesPerPage: 6,
  pageCapacity: 7, acceptedGeneration: 8, topologyGeneration: 9,
  flags: 10, faultCode: 11, dirtyPageCount: 12, omittedPageCount: 13,
  executedPageCount: 14, executedTileCount: 15, firstFaultBrick: 16,
  firstFaultTile: 17, firstFaultCause: 18, packetIndex: 19,
  indirectX: 20, indirectY: 21, indirectZ: 22,
  listBase: 23, recordsBase: 24, totalWords: 25,
  generationReceipt: 26, coverageFaultCount: 27,
  exactSampleCountLow: 28, exactSampleCountHigh: 29, publishedPageCount: 30,
  brickPagesBase: 31,
} as const);

export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE = Object.freeze({
  generation: 0, topologyGeneration: 1, flags: 2, page: 3,
  dirtyMaskLow: 4, dirtyMaskHigh: 5, executedMaskLow: 6, executedMaskHigh: 7,
  faultCode: 8, firstFaultTile: 9, causeMask: 10, logicalBrickKey: 11,
} as const);

export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG = Object.freeze({
  initialized: 1 << 0, open: 1 << 1, sealed: 1 << 2,
  localFaults: 1 << 3, globalFault: 1 << 4, executionComplete: 1 << 5,
} as const);

export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG = Object.freeze({
  scheduled: 1 << 0, omitted: 1 << 1, executing: 1 << 2,
  candidateReady: 1 << 3, localFault: 1 << 4, lifecycle: 1 << 5,
  published: 1 << 6,
} as const);

export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidFramePlan: 2, localFramePlan: 3,
  invalidPage: 4, generationMismatch: 5, unsupportedCause: 6,
  dirtyMaskMismatch: 7, lifecycleCoverage: 8, listCapacity: 9,
  uncoveredTile: 10, candidateWriteCoverage: 11,
} as const);

/** FPL1 origin/inherited cause bits accepted by the presentation packet. */
export const SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE = Object.freeze({
  topologyCreated: 1 << 0, topologyRetired: 1 << 1,
  phaseCrossing: 1 << 2, densityChanged: 1 << 3,
  boundarySource: 1 << 8, dependencyClosure: 1 << 10,
  /** Page becomes renderer-visible without requiring a topology allocation. */
  pageActivated: 1 << 13,
  /** Page becomes renderer-invisible while its physical owner remains stable. */
  pageRetired: 1 << 14,
} as const);

export interface SparseCM12FramePlanPresentationLayout {
  readonly baseWords: number;
  readonly listBaseWords: number;
  readonly brickPagesBaseWords: number;
  readonly recordsBaseWords: number;
  readonly allocatorBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly brickFineResolution: 4 | 8 | 16;
  readonly pageResolution: 4 | 8 | 16;
  readonly tilesPerPage: 1 | 8 | 64;
  /** Addressable topology leaves; unlike pageCapacity this may include dry keys. */
  readonly brickCapacity: number;
  readonly pageCapacity: number;
  readonly packetIndex: number;
  readonly indirectBinding: Readonly<{ offset: number; size: 12 }>;
}

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};
const alignWords = (value: number): number => Math.ceil(value / 64) * 64;

export function createSparseCM12FramePlanPresentationLayout(options: {
  readonly pageCapacity: number;
  readonly brickCapacity?: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly pageResolution?: 4 | 8 | 16;
  readonly packetIndex?: number;
  readonly baseWords?: number;
}): SparseCM12FramePlanPresentationLayout {
  const baseWords = alignWords(integer(options.baseWords ?? 0, "baseWords"));
  const brickFineResolution = options.brickFineResolution ?? 8;
  const pageResolution = options.pageResolution ?? brickFineResolution;
  if (pageResolution !== brickFineResolution) {
    throw new RangeError("FPP1 requires one B-sized presentation page per physical brick");
  }
  const pageCapacity = integer(options.pageCapacity, "pageCapacity");
  const brickCapacity = integer(options.brickCapacity ?? pageCapacity, "brickCapacity");
  const packetIndex = integer(options.packetIndex ?? 5, "packetIndex");
  if (packetIndex >= 6) throw new RangeError("FPP1 packetIndex must be in [0, 5]");
  const tilesPerPage = (pageResolution / 4) ** 3 as 1 | 8 | 64;
  const listBaseWords = baseWords + SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS;
  const brickPagesBaseWords = listBaseWords + pageCapacity;
  const recordsBaseWords = alignWords(brickPagesBaseWords + brickCapacity);
  const allocatorBaseWords = alignWords(recordsBaseWords
    + SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS * brickCapacity);
  const totalWords = allocatorBaseWords + 5;
  return Object.freeze({
    baseWords, listBaseWords, brickPagesBaseWords, recordsBaseWords,
    allocatorBaseWords, totalWords,
    totalBytes: 4 * totalWords, brickFineResolution, pageResolution,
    tilesPerPage, brickCapacity, pageCapacity, packetIndex,
    indirectBinding: Object.freeze({
      offset: 4 * (baseWords + SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX),
      size: 12,
    }),
  });
}

export function createSparseCM12FramePlanPresentationInitialWords(
  layout: SparseCM12FramePlanPresentationLayout,
  options: { readonly brickPages?: readonly number[] } = {},
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER;
  words[h.magic] = SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC;
  words[h.version] = SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION;
  words[h.headerWords] = SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS;
  words[h.pageWords] = SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS;
  words[h.brickFineResolution] = layout.brickFineResolution;
  words[h.pageResolution] = layout.pageResolution;
  words[h.tilesPerPage] = layout.tilesPerPage;
  words[h.pageCapacity] = layout.pageCapacity;
  words[h.packetIndex] = layout.packetIndex;
  words[h.indirectY] = 1;
  words[h.indirectZ] = 1;
  words[h.listBase] = layout.listBaseWords;
  words[h.brickPagesBase] = layout.brickPagesBaseWords;
  words[h.recordsBase] = layout.recordsBaseWords;
  words[h.totalWords] = layout.totalWords;
  words[h.firstFaultBrick] = 0xffff_ffff;
  words[h.firstFaultTile] = 0xffff_ffff;
  if (options.brickPages && options.brickPages.length !== layout.brickCapacity) {
    throw new RangeError("FPP1 brickPages must contain one entry per physical brick");
  }
  for (let brick = 0; brick < layout.brickCapacity; brick += 1) {
    words[layout.brickPagesBaseWords - layout.baseWords + brick]
      = options.brickPages?.[brick] ?? 0xffff_ffff;
    const at = layout.recordsBaseWords - layout.baseWords
      + SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS * brick;
    words[at + SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.page] = 0xffff_ffff;
    words[at + SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.firstFaultTile] = 0xffff_ffff;
  }
  const allocatedPages = new Set(options.brickPages?.filter((page) =>
    page !== 0xffff_ffff) ?? []);
  for (let page = 0; page < allocatedPages.size; page += 1) {
    if (!allocatedPages.has(page)) {
      throw new RangeError("FPP1 initial physical pages must form a compact prefix");
    }
  }
  const allocator = layout.allocatorBaseWords - layout.baseWords;
  words[allocator] = allocatedPages.size;
  words[allocator + 1] = 0;
  words[allocator + 2] = allocatedPages.size;
  words[allocator + 3] = 0;
  // Sorted prefix of the append-only renderer page directory.
  words[allocator + 4] = allocatedPages.size;
  return words;
}

export interface SparseCM12FramePlanPresentationSource {
  readonly kind: "sparse-cm12-frame-plan-presentation";
  readonly packet: GPUBufferBinding;
  readonly indirect: GPUBufferBinding;
  readonly layout: SparseCM12FramePlanPresentationLayout;
}
