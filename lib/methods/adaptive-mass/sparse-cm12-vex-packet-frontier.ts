/** Production-oriented packet frontier ABI and independent CPU oracle for VEX. */

export const SPARSE_CM12_VEX_PACKET_FRONTIER_MAGIC = 0x56585031; // VXP1
export const SPARSE_CM12_VEX_PACKET_FRONTIER_VERSION = 1;
export const SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER_WORDS = 32;
export const SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS = 6;
export const SPARSE_CM12_VEX_PACKET_FRONTIER_SCRATCH_PLANES = 17;
export const SPARSE_CM12_VEX_PACKET_FRONTIER_INVALID = 0xffff_ffff;

export const SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE = Object.freeze({
  collecting: 0,
  rootsSealed: 1,
  planning: 2,
  planned: 3,
  fault: 0xffff_ffff,
} as const);

export const SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, phase: 3,
  generation: 4, topologyGeneration: 5, topologySlot: 6, packetCapacity: 7,
  rootPacketCount: 8, rootCellCount: 9,
  frontierACandidateCount: 10, frontierAPacketCount: 11,
  frontierBCandidateCount: 12, frontierBPacketCount: 13,
  blastPacketCount: 14, blastCellCount: 15,
  faultCount: 16, firstFaultPacket: 17, firstFaultLane: 18,
  currentDepth: 19,
  rootDispatchX: 20, frontierADispatchX: 21, frontierBDispatchX: 22,
  blastDispatchX: 23,
  reserved0: 24, reserved1: 25, reserved2: 26, reserved3: 27,
  reserved4: 28, reserved5: 29, reserved6: 30, reserved7: 31,
} as const);

export interface SparseCM12VexPacketFrontierLayout {
  readonly headerBaseWords: number;
  readonly scratchBaseWords: number;
  readonly packetCapacity: number;
  readonly availableWords: number;
  readonly requiredWords: number;
  /** Root collection: generation claim stamp and stable touched-packet list. */
  readonly rootStampBaseWords: number;
  readonly rootPacketListBaseWords: number;
  /** Six vec2u cause-mask planes. Plane storage is reused only after root seal. */
  readonly rootCauseMaskLowBaseWords: readonly number[];
  readonly rootCauseMaskHighBaseWords: readonly number[];
  readonly frontierA: SparseCM12VexPacketFrontierBankLayout;
  readonly frontierB: SparseCM12VexPacketFrontierBankLayout;
  readonly blast: SparseCM12VexPacketMaskListLayout;
}

export interface SparseCM12VexPacketFrontierBankLayout {
  readonly generationStampBaseWords: number;
  readonly depthStampBaseWords: number;
  readonly candidateListBaseWords: number;
  readonly maskLowBaseWords: number;
  readonly maskHighBaseWords: number;
  readonly packetListBaseWords: number;
}

export interface SparseCM12VexPacketMaskListLayout {
  readonly stampBaseWords: number;
  readonly packetListBaseWords: number;
  readonly maskLowBaseWords: number;
  readonly maskHighBaseWords: number;
}

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

/**
 * Overlay VXP1 on the three replaceable VEX1 closure/execution-list regions
 * (frontierA/frontierB/blast). The legacy root list remains intact until every
 * producer is packet-native; it is bridged after the topology flip. No new
 * binding is required. The invariant
 * is explicit and checked: 32 + 17*packetCapacity words must fit the supplied
 * contiguous list tail. There is no alternate dense/global allocation.
 */
export function createSparseCM12VexPacketFrontierLayout(options: Readonly<{
  baseWords: number;
  availableWords: number;
  packetCapacity: number;
}>): SparseCM12VexPacketFrontierLayout {
  const headerBaseWords = integer(options.baseWords, "baseWords");
  const availableWords = integer(options.availableWords, "availableWords");
  const packetCapacity = integer(options.packetCapacity, "packetCapacity");
  const requiredWords = SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER_WORDS
    + SPARSE_CM12_VEX_PACKET_FRONTIER_SCRATCH_PLANES * packetCapacity;
  if (requiredWords > availableWords) {
    throw new RangeError(`VXP1 needs ${requiredWords} words but the VEX list overlay has ${
      availableWords}; packet frontier requires an explicit capacity replan`);
  }
  const scratchBaseWords = headerBaseWords + SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER_WORDS;
  const plane = (index: number) => scratchBaseWords + index * packetCapacity;
  // Root collection reserves plane 0 for stamps and a dedicated plane 16 for
  // the raw packet list. The list cannot alias a runtime packet-indexed plane:
  // materialization workgroups read it by rank while authoring by packet id.
  // and 12 other planes for six low/high cause masks. After sealing, the same
  // tail becomes two frontier banks (generation+depth stamps avoid token
  // aliasing across frames) plus the blast bank.
  const causePlanes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
  return Object.freeze({ headerBaseWords, scratchBaseWords, packetCapacity,
    availableWords, requiredWords,
    rootStampBaseWords: plane(0), rootPacketListBaseWords: plane(16),
    rootCauseMaskLowBaseWords: Object.freeze(
      causePlanes.filter((_, index) => (index & 1) === 0).map(plane)),
    rootCauseMaskHighBaseWords: Object.freeze(
      causePlanes.filter((_, index) => (index & 1) === 1).map(plane)),
    frontierA: Object.freeze({ generationStampBaseWords: plane(0),
      depthStampBaseWords: plane(1), candidateListBaseWords: plane(2),
      maskLowBaseWords: plane(3), maskHighBaseWords: plane(4),
      packetListBaseWords: plane(5) }),
    frontierB: Object.freeze({ generationStampBaseWords: plane(6),
      depthStampBaseWords: plane(7), candidateListBaseWords: plane(8),
      maskLowBaseWords: plane(9), maskHighBaseWords: plane(10),
      packetListBaseWords: plane(11) }),
    blast: Object.freeze({ stampBaseWords: plane(12), packetListBaseWords: plane(13),
      maskLowBaseWords: plane(14), maskHighBaseWords: plane(15) }),
  });
}

export interface SparseCM12VexPacketAddress {
  readonly packetId: number;
  readonly lane: number;
}

export interface SparseCM12VexPacketRootBatch {
  readonly generation: number;
  readonly topologyGeneration: number;
  readonly topologySlot: 0 | 1;
  readonly cause: number;
  readonly packetMasks: ReadonlyMap<number, readonly [number, number]>;
}

export interface SparseCM12VexPacketFrontierOracleResult {
  readonly rootStamp: Uint32Array;
  readonly rootCause: Uint32Array;
  readonly blastStamp: Uint32Array;
  readonly blastDepth: Uint32Array;
  readonly rootPacketMasks: ReadonlyMap<number, readonly [number, number]>;
  readonly blastPacketMasks: ReadonlyMap<number, readonly [number, number]>;
  readonly frontierPacketMasksByDepth: readonly ReadonlyMap<number,
    readonly [number, number]>[];
  readonly rootCount: number;
  readonly blastCount: number;
}

export interface SparseCM12VexPacketRootReconciliation {
  readonly batches: readonly SparseCM12VexPacketRootBatch[];
  readonly retainedPreFlipPacketCount: number;
  readonly invalidatedPreFlipPacketCount: number;
  readonly replacementRootCount: number;
}

const hasLane = (mask: readonly [number, number], lane: number): boolean =>
  (((mask[lane >>> 5]! >>> (lane & 31)) & 1) !== 0);
const setLane = (mask: [number, number], lane: number): void => {
  const word = lane >>> 5;
  mask[word] = (mask[word]! | (1 << (lane & 31))) >>> 0;
};
const frozenMasks = (masks: Map<number, [number, number]>): ReadonlyMap<number,
readonly [number, number]> => new Map([...masks].sort((a, b) => a[0] - b[0])
  .map(([packet, mask]) => [packet, Object.freeze([...mask]) as readonly [number, number]]));

/**
 * Reconcile packet roots across an accepted-selector flip without interpreting
 * an old-rung lane in the new slot. Pre-flip packets survive only when the TEI
 * compiler proves their complete packet/lane mapping unchanged. Every omitted
 * changed-leaf packet must be replaced by post-commit topology roots, checked
 * against the supplied exact candidate cell/endpoint set and cause bit.
 */
export function reconcileSparseCM12VexPacketRootBatches(options: Readonly<{
  targetTopologyGeneration: number;
  targetTopologySlot: 0 | 1;
  preFlipBatches: readonly SparseCM12VexPacketRootBatch[];
  postCommitBatches: readonly SparseCM12VexPacketRootBatch[];
  packetMappingUnchanged: (packetId: number, sourceSlot: 0 | 1,
    targetSlot: 0 | 1) => boolean;
  requiredReplacementCells: Iterable<number>;
  topologyReplacementCause: number;
  packetCell: (packetId: number, lane: number, topologySlot: 0 | 1) => number | undefined;
}>): SparseCM12VexPacketRootReconciliation {
  const targetSlot = options.targetTopologySlot;
  const retained: SparseCM12VexPacketRootBatch[] = [];
  let retainedPreFlipPacketCount = 0, invalidatedPreFlipPacketCount = 0;
  for (const batch of options.preFlipBatches) {
    if (batch.topologyGeneration > options.targetTopologyGeneration) {
      throw new Error("VXP1 pre-flip batch is newer than the target topology");
    }
    const packetMasks = new Map<number, readonly [number, number]>();
    for (const [packet, mask] of batch.packetMasks) {
      if (batch.topologySlot === targetSlot
        || options.packetMappingUnchanged(packet, batch.topologySlot, targetSlot)) {
        packetMasks.set(packet, mask); retainedPreFlipPacketCount += 1;
      } else invalidatedPreFlipPacketCount += 1;
    }
    if (packetMasks.size !== 0) retained.push(Object.freeze({ ...batch,
      topologyGeneration: options.targetTopologyGeneration,
      topologySlot: targetSlot, packetMasks }));
  }
  const replacementCauseByCell = new Map<number, number>();
  for (const batch of options.postCommitBatches) {
    if (batch.topologyGeneration !== options.targetTopologyGeneration
      || batch.topologySlot !== targetSlot) {
      throw new Error("VXP1 post-commit root batch does not name the frozen target slot");
    }
    retained.push(batch);
    for (const [packet, mask] of batch.packetMasks) for (let lane = 0; lane < 64; lane += 1) {
      if (!hasLane(mask, lane)) continue;
      const cell = options.packetCell(packet, lane, targetSlot);
      if (cell === undefined) throw new Error(`VXP1 replacement selects invalid ${packet}/${lane}`);
      replacementCauseByCell.set(cell,
        (replacementCauseByCell.get(cell) ?? 0) | batch.cause);
    }
  }
  let replacementRootCount = 0;
  for (const cell of new Set(options.requiredReplacementCells)) {
    replacementRootCount += 1;
    if (((replacementCauseByCell.get(cell) ?? 0) & options.topologyReplacementCause) === 0) {
      throw new Error(`VXP1 changed topology cell/endpoint ${cell} lacks post-commit replacement root`);
    }
  }
  return Object.freeze({ batches: Object.freeze(retained), retainedPreFlipPacketCount,
    invalidatedPreFlipPacketCount, replacementRootCount });
}

/**
 * Independent set oracle for the packetized depth-zero merge and eight-hop
 * recurrence. It deliberately uses cell-neighbor traversal rather than the
 * shift/mask implementation used by the GPU transform.
 */
export function executeSparseCM12VexPacketFrontierOracle(options: Readonly<{
  cellCapacity: number;
  packetCapacity: number;
  batches: readonly SparseCM12VexPacketRootBatch[];
  packetCell: (packetId: number, lane: number, topologySlot: 0 | 1) => number | undefined;
  cellAddress: (cell: number, topologySlot: 0 | 1) => SparseCM12VexPacketAddress | undefined;
  cellActive: (cell: number, topologySlot: 0 | 1) => boolean;
  neighbors: (cell: number, topologySlot: 0 | 1) => Iterable<number>;
  maximumDepth?: number;
}>): SparseCM12VexPacketFrontierOracleResult {
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const packetCapacity = integer(options.packetCapacity, "packetCapacity");
  const maximumDepth = integer(options.maximumDepth ?? 8, "maximumDepth");
  if (options.batches.length === 0) throw new RangeError("VXP1 oracle needs a root batch");
  const first = options.batches[0]!;
  if (first.generation === 0 || first.generation >= 0x7fff_fffe) {
    throw new RangeError("VXP1 generation must be in [1, 0x7ffffffd]");
  }
  const generation = first.generation, topologyGeneration = first.topologyGeneration;
  const topologySlot = first.topologySlot;
  const rootStamp = new Uint32Array(cellCapacity);
  const rootCause = new Uint32Array(cellCapacity);
  const blastStamp = new Uint32Array(cellCapacity);
  const blastDepth = new Uint32Array(cellCapacity).fill(SPARSE_CM12_VEX_PACKET_FRONTIER_INVALID);
  const rootMasks = new Map<number, [number, number]>();
  for (const batch of options.batches) {
    if (batch.generation !== generation || batch.topologyGeneration !== topologyGeneration
      || batch.topologySlot !== topologySlot) {
      throw new Error("VXP1 cannot merge batches from different generation/TEI slots");
    }
    if (!Number.isSafeInteger(batch.cause) || batch.cause <= 0
      || (batch.cause & ~0x3f) !== 0) throw new RangeError("VXP1 cause must use bits [0, 5]");
    for (const [packetId, mask] of batch.packetMasks) {
      if (packetId < 0 || packetId >= packetCapacity) {
        throw new RangeError(`VXP1 packet ${packetId} exceeds capacity`);
      }
      const merged = rootMasks.get(packetId) ?? [0, 0] as [number, number];
      for (let lane = 0; lane < 64; lane += 1) {
        if (!hasLane(mask, lane)) continue;
        const cell = options.packetCell(packetId, lane, topologySlot);
        if (cell === undefined || cell < 0 || cell >= cellCapacity
          || !options.cellActive(cell, topologySlot)) {
          throw new Error(`VXP1 batch selects invalid packet lane ${packetId}/${lane}`);
        }
        setLane(merged, lane); rootStamp[cell] = generation;
        rootCause[cell] = (rootCause[cell]! | batch.cause) >>> 0;
      }
      rootMasks.set(packetId, merged);
    }
  }
  const frontiers: Map<number, [number, number]>[] = [new Map()];
  for (const [packet, mask] of rootMasks) frontiers[0]!.set(packet, [...mask]);
  const blastMasks = new Map<number, [number, number]>();
  let rootCount = 0, blastCount = 0;
  for (const [packet, mask] of rootMasks) {
    blastMasks.set(packet, [...mask]);
    for (let lane = 0; lane < 64; lane += 1) if (hasLane(mask, lane)) {
      const cell = options.packetCell(packet, lane, topologySlot)!;
      blastStamp[cell] = generation; blastDepth[cell] = 0;
      rootCount += 1; blastCount += 1;
    }
  }
  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    const next = new Map<number, [number, number]>();
    for (const [packet, mask] of frontiers[depth - 1]!) {
      for (let lane = 0; lane < 64; lane += 1) {
        if (!hasLane(mask, lane)) continue;
        const cell = options.packetCell(packet, lane, topologySlot)!;
        for (const neighbor of options.neighbors(cell, topologySlot)) {
          if (neighbor < 0 || neighbor >= cellCapacity
            || !options.cellActive(neighbor, topologySlot)
            || blastStamp[neighbor] === generation) continue;
          const target = options.cellAddress(neighbor, topologySlot);
          if (!target || target.packetId >= packetCapacity || target.lane >= 64) {
            throw new Error(`VXP1 active cell ${neighbor} has no accepted TEI packet lane`);
          }
          let targetMask = next.get(target.packetId);
          if (!targetMask) next.set(target.packetId, targetMask = [0, 0]);
          setLane(targetMask, target.lane);
          blastStamp[neighbor] = generation; blastDepth[neighbor] = depth;
          let blastMask = blastMasks.get(target.packetId);
          if (!blastMask) blastMasks.set(target.packetId, blastMask = [0, 0]);
          setLane(blastMask, target.lane); blastCount += 1;
        }
      }
    }
    frontiers.push(next);
  }
  return Object.freeze({ rootStamp, rootCause, blastStamp, blastDepth,
    rootPacketMasks: frozenMasks(rootMasks), blastPacketMasks: frozenMasks(blastMasks),
    frontierPacketMasksByDepth: Object.freeze(frontiers.map(frozenMasks)),
    rootCount, blastCount });
}

/**
 * Execute the production-shaped recurrence using packet masks only. The
 * supplied transform is the exact compiled local operator (it may include
 * identity; visited subtraction removes it). This path never builds a cell
 * root, frontier, or blast list.
 */
export function executeSparseCM12VexDirectPacketRecurrence(options: Readonly<{
  cellCapacity: number;
  packetCapacity: number;
  batches: readonly SparseCM12VexPacketRootBatch[];
  packetCell: (packetId: number, lane: number, topologySlot: 0 | 1) => number | undefined;
  cellActive: (cell: number, topologySlot: 0 | 1) => boolean;
  expandPacketMask: (packetId: number, low: number, high: number) =>
    ReadonlyMap<number, readonly [number, number]>;
  maximumDepth?: number;
}>): SparseCM12VexPacketFrontierOracleResult {
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const packetCapacity = integer(options.packetCapacity, "packetCapacity");
  const maximumDepth = integer(options.maximumDepth ?? 8, "maximumDepth");
  if (options.batches.length === 0) throw new RangeError("VXP1 recurrence needs a root batch");
  const first = options.batches[0]!, generation = first.generation;
  const topologySlot = first.topologySlot;
  const rootStamp = new Uint32Array(cellCapacity), rootCause = new Uint32Array(cellCapacity);
  const blastStamp = new Uint32Array(cellCapacity);
  const blastDepth = new Uint32Array(cellCapacity).fill(SPARSE_CM12_VEX_PACKET_FRONTIER_INVALID);
  const rootMasks = new Map<number, [number, number]>();
  for (const batch of options.batches) {
    if (batch.generation !== generation || batch.topologyGeneration !== first.topologyGeneration
      || batch.topologySlot !== topologySlot) {
      throw new Error("VXP1 cannot merge batches from different generation/TEI slots");
    }
    for (const [packet, mask] of batch.packetMasks) {
      if (packet < 0 || packet >= packetCapacity) throw new RangeError("VXP1 packet exceeds capacity");
      let merged = rootMasks.get(packet);if (!merged) rootMasks.set(packet, merged = [0, 0]);
      merged[0] = (merged[0] | mask[0]) >>> 0;merged[1] = (merged[1] | mask[1]) >>> 0;
      for (let lane = 0; lane < 64; lane += 1) if (hasLane(mask, lane)) {
        const cell = options.packetCell(packet, lane, topologySlot);
        if (cell === undefined || !options.cellActive(cell, topologySlot)) {
          throw new Error(`VXP1 batch selects invalid packet lane ${packet}/${lane}`);
        }
        rootStamp[cell] = generation;rootCause[cell] = (rootCause[cell]! | batch.cause) >>> 0;
      }
    }
  }
  let frontier = new Map<number, [number, number]>([...rootMasks]
    .map(([packet, mask]) => [packet, [...mask] as [number, number]]));
  const frontiers: Map<number, [number, number]>[] = [frontier];
  const blastMasks = new Map<number, [number, number]>();
  let rootCount = 0, blastCount = 0;
  for (const [packet, mask] of frontier) {
    blastMasks.set(packet, [...mask]);
    for (let lane = 0; lane < 64; lane += 1) if (hasLane(mask, lane)) {
      const cell = options.packetCell(packet, lane, topologySlot)!;
      blastStamp[cell] = generation;blastDepth[cell] = 0;rootCount += 1;blastCount += 1;
    }
  }
  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    const candidates = new Map<number, [number, number]>();
    for (const [sourcePacket, sourceMask] of frontier) {
      for (const [targetPacket, targetMask] of options.expandPacketMask(
        sourcePacket, sourceMask[0], sourceMask[1])) {
        let merged = candidates.get(targetPacket);
        if (!merged) candidates.set(targetPacket, merged = [0, 0]);
        merged[0] = (merged[0] | targetMask[0]) >>> 0;
        merged[1] = (merged[1] | targetMask[1]) >>> 0;
      }
    }
    const next = new Map<number, [number, number]>();
    for (const [packet, candidate] of candidates) {
      let blast = blastMasks.get(packet);if (!blast) blastMasks.set(packet, blast = [0, 0]);
      const novel: [number, number] = [
        (candidate[0] & ~blast[0]) >>> 0, (candidate[1] & ~blast[1]) >>> 0,
      ];
      if ((novel[0] | novel[1]) === 0) continue;
      next.set(packet, novel);blast[0] = (blast[0] | novel[0]) >>> 0;
      blast[1] = (blast[1] | novel[1]) >>> 0;
      for (let lane = 0; lane < 64; lane += 1) if (hasLane(novel, lane)) {
        const cell = options.packetCell(packet, lane, topologySlot);
        if (cell === undefined || !options.cellActive(cell, topologySlot)) {
          throw new Error(`VXP1 expansion selects invalid packet lane ${packet}/${lane}`);
        }
        blastStamp[cell] = generation;blastDepth[cell] = depth;blastCount += 1;
      }
    }
    frontier = next;frontiers.push(frontier);
  }
  return Object.freeze({ rootStamp, rootCause, blastStamp, blastDepth,
    rootPacketMasks: frozenMasks(rootMasks), blastPacketMasks: frozenMasks(blastMasks),
    frontierPacketMasksByDepth: Object.freeze(frontiers.map(frozenMasks)),
    rootCount, blastCount });
}
