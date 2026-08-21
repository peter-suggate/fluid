/** B16-native, double-buffered GPU frame scheduling and dirty-overlay ABI. */
export const SPARSE_CM12_FRAME_PLAN_MAGIC = 0x4650_4c31; // FPL1
export const SPARSE_CM12_FRAME_PLAN_VERSION = 1;
export const SPARSE_CM12_FRAME_PLAN_HEADER_WORDS = 64;
export const SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS = 32;
export const SPARSE_CM12_FRAME_PLAN_BRICK_WORDS = 32;
export const SPARSE_CM12_FRAME_PLAN_TILE_WORDS = 4; // 16-byte hot record
export const SPARSE_CM12_FRAME_PLAN_STAGE_COUNT = 6;
export const SPARSE_CM12_FRAME_PLAN_TILE_EDGE = 4;
export const SPARSE_CM12_FRAME_PLAN_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_FRAME_PLAN_INVALID = 0xffff_ffff;
export const SPARSE_CM12_FRAME_PLAN_INDIRECT_WORDS =
  3 * SPARSE_CM12_FRAME_PLAN_STAGE_COUNT;

export const SPARSE_CM12_FRAME_PLAN_STAGE = Object.freeze({
  facePreparation: 0,
  massTransport: 1,
  gammaTransport: 2,
  surfaceConditioning: 3,
  pressureCoefficients: 4,
  presentation: 5,
} as const);

export const SPARSE_CM12_FRAME_PLAN_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, slotHeaderWords: 3,
  brickWords: 4, tileWords: 5, stageCount: 6, tileEdge: 7,
  brickFineResolution: 8, tilesPerAxis: 9, tilesPerBrick: 10, brickCapacity: 11,
  currentSlot: 12, previousSlot: 13, acceptedGeneration: 14, candidateGeneration: 15,
  flags: 16, faultCode: 17, firstFaultBrick: 18, firstFaultTile: 19,
  firstFaultStage: 20, acceptedTopologyGeneration: 21,
  candidateTopologyGeneration: 22, consumerGeneration: 23,
  slot0Base: 24, slot1Base: 25, slotWords: 26, totalWords: 27,
  indirectWords: 28, packetCount: 29,
  /** Physics ping-pong authority; CPU uploads dt/forces but never field parity. */
  acceptedFrameGeneration: 30, acceptedParity: 31,
  /** Six census words, then six tightly packed vec3 indirect triplets. */
  fixedPacketCounts: 32, fixedIndirectArguments: 38,
} as const);

export const SPARSE_CM12_FRAME_PLAN_SLOT = Object.freeze({
  generation: 0, topologyGeneration: 1, flags: 2, localFaultBrickCount: 3,
  firstFaultBrick: 4, sealedBrickCount: 5, executedBrickCount: 6, reserved: 7,
  /** Four words per physical packet: brick count then an indirect xyz triplet. */
  stagePackets: 8,
} as const);

export const SPARSE_CM12_FRAME_PLAN_BRICK = Object.freeze({
  generation: 0, topologyGeneration: 1, flags: 2, logicalBrickKey: 3,
  stageMasks: 4, // six low/high pairs
  packedPacketIndicesLow: 16, packedPacketIndicesHigh: 17,
  causeMask: 18, packedClosureDepths: 19,
  executedStageMask: 20, skippedStageMask: 21,
  producerGeneration: 22, consumerGeneration: 23,
  faultCode: 24, firstFaultTile: 25, firstFaultStage: 26,
  expectedStageMask: 27, validTileMaskLow: 28, validTileMaskHigh: 29,
  packetEpoch: 30, reserved: 31,
} as const);

export const SPARSE_CM12_FRAME_PLAN_TILE = Object.freeze({
  generation: 0,
  /** Six-bit direct, closure, executed, skipped, and uncovered lanes. */
  packedStageMasks: 1,
  /** Low 16 origin causes; high 16 inherited causes. */
  packedCauseMasks: 2,
  /** Six four-bit closure-depth lanes; high eight bits reserved. */
  packedClosureDepths: 3,
} as const);

export const SPARSE_CM12_FRAME_PLAN_FLAG = Object.freeze({
  initialized: 1 << 0, candidateOpen: 1 << 1, candidateSealed: 1 << 2,
  accepted: 1 << 3, globalFault: 1 << 4, localFaults: 1 << 5,
} as const);

export const SPARSE_CM12_FRAME_PLAN_SLOT_FLAG = Object.freeze({
  open: 1 << 0, brickSealed: 1 << 1, indirectSealed: 1 << 2,
  accepted: 1 << 3, localFaults: 1 << 4,
} as const);

export const SPARSE_CM12_FRAME_PLAN_BRICK_FLAG = Object.freeze({
  initialized: 1 << 0, sealed: 1 << 1, localFault: 1 << 2,
  hasWork: 1 << 3, coverageComplete: 1 << 4,
} as const);

export const SPARSE_CM12_FRAME_PLAN_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, generationMismatch: 2, invalidBrick: 3,
  invalidTile: 4, invalidStage: 5, invalidPacket: 6, localGeneration: 7,
  missingCoverage: 8, listCapacity: 9, generationExhausted: 10,
  invalidTransition: 11,
} as const);

export interface SparseCM12FramePlanLayout {
  readonly baseWords: number;
  readonly slot0BaseWords: number;
  readonly slot1BaseWords: number;
  readonly slotWords: number;
  readonly brickBaseOffsetWords: number;
  readonly tileBaseOffsetWords: number;
  readonly stageListBaseOffsetWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly brickFineResolution: 4 | 8 | 16;
  readonly tilesPerAxis: 1 | 2 | 4;
  readonly tilesPerBrick: 1 | 8 | 64;
  readonly brickCapacity: number;
  readonly packetCount: number;
  readonly indirectSnapshotWords: number;
  readonly overlayBinding: Readonly<{ offset: number; size: number }>;
  readonly fixedIndirectBinding: Readonly<{ offset: number; size: number }>;
}

export interface SparseCM12FramePlanSource {
  readonly kind: "sparse-cm12-frame-plan";
  readonly plan: GPUBufferBinding;
  /** Fixed GPU-authored source range copied at the storage/indirect seam. */
  readonly indirectSource: GPUBufferBinding;
  /** Copy-isolated physical-packet indirect dispatch triplets. */
  readonly indirect: GPUBufferBinding;
  readonly layout: SparseCM12FramePlanLayout;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

const alignWords = (value: number): number => Math.ceil(
  value / SPARSE_CM12_FRAME_PLAN_ALIGNMENT_WORDS,
) * SPARSE_CM12_FRAME_PLAN_ALIGNMENT_WORDS;

export function sparseCM12FramePlanValidTileMask(
  brickFineResolution: 4 | 8 | 16,
): readonly [number, number] {
  const count = (brickFineResolution / SPARSE_CM12_FRAME_PLAN_TILE_EDGE) ** 3;
  if (count < 32) return [(2 ** count - 1) >>> 0, 0] as const;
  if (count === 32) return [0xffff_ffff, 0] as const;
  return [0xffff_ffff, 0xffff_ffff] as const;
}

export function sparseCM12FramePlanIndirectByteOffset(packet: number): number {
  if (!Number.isSafeInteger(packet) || packet < 0
    || packet >= SPARSE_CM12_FRAME_PLAN_STAGE_COUNT) {
    throw new RangeError("FramePlan packet must be in [0, 5]");
  }
  return 12 * packet;
}

export function createSparseCM12FramePlanLayout(options: {
  readonly brickCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly packetCount?: number;
  readonly baseWords?: number;
}): SparseCM12FramePlanLayout {
  const baseWords = checked(options.baseWords ?? 0, "baseWords");
  if (baseWords % SPARSE_CM12_FRAME_PLAN_ALIGNMENT_WORDS !== 0) {
    throw new RangeError("FramePlan baseWords must be 256-byte aligned");
  }
  const brickFineResolution = options.brickFineResolution ?? 8;
  const tilesPerAxis = (brickFineResolution / SPARSE_CM12_FRAME_PLAN_TILE_EDGE) as 1 | 2 | 4;
  const tilesPerBrick = tilesPerAxis ** 3 as 1 | 8 | 64;
  const brickCapacity = checked(options.brickCapacity, "brickCapacity");
  const packetCount = checked(options.packetCount ?? SPARSE_CM12_FRAME_PLAN_STAGE_COUNT,
    "packetCount");
  if (packetCount < 1 || packetCount > SPARSE_CM12_FRAME_PLAN_STAGE_COUNT) {
    throw new RangeError("FramePlan packetCount must be in [1, 6]");
  }
  const brickBaseOffsetWords = SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS;
  const tileBaseOffsetWords = brickBaseOffsetWords
    + SPARSE_CM12_FRAME_PLAN_BRICK_WORDS * brickCapacity;
  const stageListBaseOffsetWords = tileBaseOffsetWords
    + SPARSE_CM12_FRAME_PLAN_TILE_WORDS * tilesPerBrick * brickCapacity;
  const slotWords = alignWords(stageListBaseOffsetWords
    + packetCount * brickCapacity);
  const slot0BaseWords = baseWords + SPARSE_CM12_FRAME_PLAN_HEADER_WORDS;
  const slot1BaseWords = slot0BaseWords + slotWords;
  const totalWords = slot1BaseWords + slotWords;
  if (!Number.isSafeInteger(totalWords) || totalWords > 0x3fff_ffff) {
    throw new RangeError("FramePlan exceeds the addressable u32 GPU arena");
  }
  return Object.freeze({
    baseWords, slot0BaseWords, slot1BaseWords, slotWords,
    brickBaseOffsetWords, tileBaseOffsetWords, stageListBaseOffsetWords,
    totalWords, totalBytes: 4 * totalWords,
    brickFineResolution, tilesPerAxis, tilesPerBrick, brickCapacity, packetCount,
    indirectSnapshotWords: 3 * packetCount,
    // WGSL offsets remain absolute when FramePlan is appended to a shared
    // arena, so native scheduler/overlay consumers bind the complete prefix.
    overlayBinding: Object.freeze({ offset: 0, size: 4 * totalWords }),
    fixedIndirectBinding: Object.freeze({
      offset: 4 * (baseWords + SPARSE_CM12_FRAME_PLAN_HEADER.fixedIndirectArguments),
      size: 12 * packetCount,
    }),
  });
}

export function createSparseCM12FramePlanInitialWords(
  layout: SparseCM12FramePlanLayout,
): Uint32Array {
  const result = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FRAME_PLAN_HEADER;
  result[h.magic] = SPARSE_CM12_FRAME_PLAN_MAGIC;
  result[h.version] = SPARSE_CM12_FRAME_PLAN_VERSION;
  result[h.headerWords] = SPARSE_CM12_FRAME_PLAN_HEADER_WORDS;
  result[h.slotHeaderWords] = SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS;
  result[h.brickWords] = SPARSE_CM12_FRAME_PLAN_BRICK_WORDS;
  result[h.tileWords] = SPARSE_CM12_FRAME_PLAN_TILE_WORDS;
  result[h.stageCount] = SPARSE_CM12_FRAME_PLAN_STAGE_COUNT;
  result[h.tileEdge] = SPARSE_CM12_FRAME_PLAN_TILE_EDGE;
  result[h.brickFineResolution] = layout.brickFineResolution;
  result[h.tilesPerAxis] = layout.tilesPerAxis;
  result[h.tilesPerBrick] = layout.tilesPerBrick;
  result[h.brickCapacity] = layout.brickCapacity;
  result[h.currentSlot] = 0;
  result[h.previousSlot] = 1;
  result[h.flags] = SPARSE_CM12_FRAME_PLAN_FLAG.initialized;
  result[h.firstFaultBrick] = SPARSE_CM12_FRAME_PLAN_INVALID;
  result[h.firstFaultTile] = SPARSE_CM12_FRAME_PLAN_INVALID;
  result[h.firstFaultStage] = SPARSE_CM12_FRAME_PLAN_INVALID;
  result[h.slot0Base] = layout.slot0BaseWords;
  result[h.slot1Base] = layout.slot1BaseWords;
  result[h.slotWords] = layout.slotWords;
  result[h.totalWords] = layout.totalWords;
  result[h.indirectWords] = layout.indirectSnapshotWords;
  result[h.packetCount] = layout.packetCount;
  result[h.acceptedFrameGeneration] = 0;
  result[h.acceptedParity] = 0;
  for (let packet = 0; packet < layout.packetCount; packet += 1) {
    const at = h.fixedIndirectArguments + 3 * packet;
    result[at + 1] = 1;
    result[at + 2] = 1;
  }
  for (const slotBase of [layout.slot0BaseWords, layout.slot1BaseWords]) {
    const relative = slotBase - layout.baseWords;
    result[relative + SPARSE_CM12_FRAME_PLAN_SLOT.firstFaultBrick]
      = SPARSE_CM12_FRAME_PLAN_INVALID;
    for (let packetIndex = 0; packetIndex < layout.packetCount; packetIndex += 1) {
      const packet = relative + SPARSE_CM12_FRAME_PLAN_SLOT.stagePackets + 4 * packetIndex;
      result[packet + 2] = 1;
      result[packet + 3] = 1;
    }
  }
  const [validLow, validHigh] = sparseCM12FramePlanValidTileMask(
    layout.brickFineResolution,
  );
  for (const slotBase of [layout.slot0BaseWords, layout.slot1BaseWords]) {
    for (let brick = 0; brick < layout.brickCapacity; brick += 1) {
      const at = slotBase + layout.brickBaseOffsetWords
        + SPARSE_CM12_FRAME_PLAN_BRICK_WORDS * brick - layout.baseWords;
      result[at + SPARSE_CM12_FRAME_PLAN_BRICK.logicalBrickKey]
        = SPARSE_CM12_FRAME_PLAN_INVALID;
      result[at + SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultTile]
        = SPARSE_CM12_FRAME_PLAN_INVALID;
      result[at + SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultStage]
        = SPARSE_CM12_FRAME_PLAN_INVALID;
      result[at + SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskLow] = validLow;
      result[at + SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskHigh] = validHigh;
    }
  }
  return result;
}

export function sparseCM12FramePlanSource(
  layout: SparseCM12FramePlanLayout,
  plan: GPUBuffer,
  indirect: GPUBuffer,
): SparseCM12FramePlanSource {
  return Object.freeze({
    kind: "sparse-cm12-frame-plan" as const,
    plan: { buffer: plan, offset: layout.overlayBinding.offset,
      size: layout.overlayBinding.size },
    indirectSource: { buffer: plan, offset: layout.fixedIndirectBinding.offset,
      size: layout.fixedIndirectBinding.size },
    indirect: { buffer: indirect, offset: 0,
      size: 4 * layout.indirectSnapshotWords },
    layout,
  });
}
