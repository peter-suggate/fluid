import type { SparseAtlasCompositeGrid } from "../sparse-atlas-composite-projection";
import {
  sparseBrickLadder,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
} from "../sparse-brick-atlas";
import {
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_HEADER,
  createSparseCM12HotTopology,
  type SparseCM12HotTopology,
} from "../sparse-cm12-hot-topology";
import {
  SPARSE_CM12_LOGICAL_OWNER_HEADER,
  SPARSE_CM12_LOGICAL_OWNER_RECORD,
  SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS,
  createSparseCM12LogicalOwnerDirectory,
  type SparseCM12LogicalOwnerRuntime,
} from "../sparse-cm12-logical-owner-directory";
import {
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
  createSparseCM12TransportExecutionImage,
  createSparseCM12TransportExecutionImageLayout,
  type SparseCM12TransportExecutionImage,
  type SparseCM12TransportExecutionImageLayout,
} from "../sparse-cm12-transport-execution-image";
import {
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY,
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS,
  SPARSE_CM12_WORLD_DIRECTORY_HEADER,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS,
  createSparseCM12WorldDirectoryInitialWords,
  createSparseCM12WorldDirectoryLayout,
  sparseCM12WorldCoordinateHash,
  type SparseCM12WorldDirectoryLayout,
} from "../sparse-cm12-world-directory";
import type { SliceTopology } from "./slice-topology";

const INVALID = 0xffff_ffff;

export type SliceRuntimeAuthorityPhase = "accepted" | "candidate-ready";

/**
 * One Z-reduced topology encoded by the production WDR1, TEI2 and HTP1
 * constructors. Z has physical extent one and only TEI lane z=0 is live; no
 * record format, header phase, stable packet stride, or row ABI is replaced.
 */
export interface SliceRuntimeImage {
  readonly topology: SliceTopology;
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  readonly worldDirectoryLayout: SparseCM12WorldDirectoryLayout;
  readonly worldDirectoryWords: Uint32Array;
  readonly transportExecutionImage: SparseCM12TransportExecutionImage;
  readonly hotTopology: SparseCM12HotTopology;
  /** Dense execution indices in the accepted HTP cell stream. */
  readonly cellInvocationOrder: Uint32Array;
  /** Stable B8 cell addresses (`leaf * 64 + local`) for field-bank access. */
  readonly stableCellOrder: Uint32Array;
  /** Canonical HTP row invocation stream. */
  readonly rowInvocationOrder: Uint32Array;
  /** Stable TEI packet ids; invalid/spare packet records are omitted. */
  readonly transportPacketSchedule: Uint32Array;
  /** Compact constructor slot to stable physical production leaf id. */
  readonly compactLeafToStable: Uint32Array;
}

export interface SliceRuntimeAuthorityReceipt {
  readonly phase: SliceRuntimeAuthorityPhase;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly acceptedLeafCount: number;
  readonly candidateLeafCount: number;
  readonly acceptedCellCount: number;
  readonly candidateCellCount: number;
  readonly acceptedRowCount: number;
  readonly candidateRowCount: number;
  readonly acceptedPacketCount: number;
  readonly candidatePacketCount: number;
  readonly acceptedSlot: 0 | 1;
  readonly candidateSlot: 0 | 1;
  readonly fault: number;
  readonly firstFaultId: number;
}

export interface SliceRuntimeAuthority {
  readonly leafCapacity: number;
  readonly accepted: SliceRuntimeImage;
  readonly candidate?: SliceRuntimeImage;
  /** Production TEI2 double bank: accepted and candidate occupy opposite slots. */
  readonly transportExecutionWords: Uint32Array;
  readonly acceptedSlot: 0 | 1;
  readonly receipt: SliceRuntimeAuthorityReceipt;
}

export interface SliceRuntimeAuthorityOptions {
  /** Fixed WDR/TEI arena capacity. Runtime page growth consumes this tail. */
  readonly leafCapacity?: number;
}

export interface SliceRuntimeReleaseReceipt {
  readonly requestedLeafIds: Uint32Array;
  readonly releasedLeafIds: Uint32Array;
  /** Valid dynamic IDs that production's release predicate declined. */
  readonly rejectedLeafIds: Uint32Array;
  readonly generation: number;
  readonly liveLeafCount: number;
  readonly freeLeafCount: number;
  readonly nextLeaf: number;
  readonly boundsGeneration: number;
  /** WDR capacity/free-list corruption; ordinary rejected IDs are not faults. */
  readonly fault: number;
  readonly firstFaultId: number;
}

export interface SliceRuntimeReleaseResult {
  readonly authority: SliceRuntimeAuthority;
  readonly receipt: SliceRuntimeReleaseReceipt;
}

const brickDimensions = (topology: SliceTopology): readonly [number, number, number] =>
  [Math.ceil(topology.dimensions[0] / topology.brickFineResolution),
    Math.ceil(topology.dimensions[1] / topology.brickFineResolution), 1];

function requireStableLeafSlots(topology: SliceTopology, leafCapacity: number): void {
  const ids = new Set<number>();
  topology.bricks.forEach(brick => {
    if (!Number.isSafeInteger(brick.id) || brick.id < 0 || ids.has(brick.id)) {
      throw new Error(`slice WDR has invalid or duplicate stable leaf slot ${brick.id}`);
    }
    ids.add(brick.id);
  });
  if (topology.bricks.length > leafCapacity) {
    throw new RangeError(`slice WDR leaf capacity ${leafCapacity} cannot hold ${topology.bricks.length}`);
  }
  const highWater = Math.max(0, ...topology.bricks.map(brick => brick.id + 1));
  if (highWater > leafCapacity) {
    throw new RangeError(`slice WDR stable leaf high-water ${highWater} exceeds capacity ${leafCapacity}`);
  }
}

function productionAtlas(topology: SliceTopology): SparseAdaptiveMassAtlas {
  const dimensions = [topology.dimensions[0], topology.dimensions[1], 1] as const;
  const logical = brickDimensions(topology);
  const bricks: SparseAdaptiveMassBrick[] = topology.bricks.map(source => {
    const count = source.resolution ** 3;
    const density = new Float64Array(count), gamma = new Float64Array(count);
    // Payload is not consumed by WDR/TEI/HTP construction. Still reproduce a
    // valid one-layer extrusion so source-derived validators may inspect it.
    for (let z = 0; z < source.resolution; z += 1) {
      for (let y = 0; y < source.resolution; y += 1) {
        for (let x = 0; x < source.resolution; x += 1) {
          const destination = x + source.resolution * (y + source.resolution * z);
          const sourceIndex = x + source.resolution * y;
          density[destination] = source.density?.[sourceIndex] ?? 0;
          gamma[destination] = source.gamma?.[sourceIndex] ?? 0;
        }
      }
    }
    const projectedKey = source.coordinate[0] + logical[0] * source.coordinate[1];
    return Object.freeze({ key: projectedKey,
      coordinate: [source.coordinate[0], source.coordinate[1], 0] as const,
      ...(source.spanBricks === undefined ? {} : { spanBricks: source.spanBricks }),
      resolution: source.resolution, density, gamma });
  });
  const directory = new Map(bricks.map(brick => [brick.key, brick]));
  const directoriesBySpan = new Map<number, ReadonlyMap<number, SparseAdaptiveMassBrick>>();
  for (const brick of bricks) {
    const span = brick.spanBricks ?? 1;
    let byKey = directoriesBySpan.get(span) as Map<number, SparseAdaptiveMassBrick> | undefined;
    if (!byKey) directoriesBySpan.set(span, byKey = new Map());
    byKey.set(brick.key, brick);
  }
  const maximumSpanBricks = Math.max(1, ...bricks.map(brick => brick.spanBricks ?? 1));
  return Object.freeze({ signedCoordinates: false, dimensions,
    brickFineResolution: 8, brickCellCapacity: 8 ** 3, ladder: sparseBrickLadder(8),
    brickDimensions: brickDimensions(topology), bricks: Object.freeze(bricks), directory,
    directoriesBySpan, maximumSpanBricks, generation: topology.generation });
}

function productionGrid(topology: SliceTopology,
  atlas: SparseAdaptiveMassAtlas): SparseAtlasCompositeGrid {
  const internalKey = new Map(topology.bricks.map((brick, compact) =>
    [brick.key, atlas.bricks[compact]!.key]));
  const internalCellBase = new Map<number, number>();
  for (const [sourceKey, base] of topology.cellBaseByBrick) {
    internalCellBase.set(internalKey.get(sourceKey)!, base);
  }
  return Object.freeze({ atlas,
    cells: Object.freeze(topology.cells.map(cell => Object.freeze({ id: cell.id,
      stableLeafId: cell.stableLeafId, brickKey: internalKey.get(cell.brickKey)!,
      brickCoordinate: [cell.brickCoordinate[0], cell.brickCoordinate[1], 0] as const,
      brickResolution: cell.brickResolution,
      local: [cell.local[0], cell.local[1], 0] as const, localIndex: cell.localIndex,
      minimumFine: [cell.minimumFine[0], cell.minimumFine[1], 0] as const,
      maximumFine: [cell.maximumFine[0], cell.maximumFine[1], 1] as const,
      centerFine: [cell.centerFine[0], cell.centerFine[1], 0.5] as const,
      widthsFine: [cell.widthsFine[0], cell.widthsFine[1], 1] as const,
      volume: cell.volumeFineCells, volumeFineCells: cell.volumeFineCells,
      density: cell.density, gamma: cell.gamma }))),
    gradientRows: Object.freeze(topology.rows.map(row => Object.freeze({ id: row.id,
      kind: row.kind, axis: row.axis,
      centerFine: [row.centerFine[0], row.centerFine[1], 0.5] as const,
      area: row.areaFineCells, distance: row.centerDistanceFine,
      areaFineCells2: row.areaFineCells, centerDistanceFine: row.centerDistanceFine,
      dualWeight: row.dualWeight, terms: row.terms,
      ...(row.negativeBrickKey === undefined ? {}
        : { negativeBrickKey: internalKey.get(row.negativeBrickKey)! }),
      ...(row.positiveBrickKey === undefined ? {}
        : { positiveBrickKey: internalKey.get(row.positiveBrickKey)! }),
      ...(row.exteriorPhi === undefined ? {} : { exteriorPhi: row.exteriorPhi }) }))),
    cellBaseByBrick: internalCellBase,
    mixedSeamRowCount: topology.mixedSeamRowCount,
    sparseAirRowCount: topology.sparseAirRowCount,
    topologyKey: topology });
}

function logicalSlotsPerLeaf(topology: SliceTopology): number {
  const logical = brickDimensions(topology);
  return Math.max(1, ...topology.bricks.map(brick => {
    const span = brick.spanBricks ?? 1;
    const x = Math.max(0, Math.min(span, logical[0] - brick.coordinate[0]));
    const y = Math.max(0, Math.min(span, logical[1] - brick.coordinate[1]));
    // This is production's row-major highest occupied logical-slot index + 1,
    // specialized only by the one-deep Z extent.
    return (y - 1) * span + x;
  }));
}

function transportLayout(topology: SliceTopology,
  leafCapacity: number): SparseCM12TransportExecutionImageLayout {
  return createSparseCM12TransportExecutionImageLayout({ brickFineResolution: 8,
    logicalBrickDimensions: brickDimensions(topology), leafCapacity,
    maximumSpanBricks: Math.max(1,
      ...topology.bricks.map(brick => brick.spanBricks ?? 1)),
    logicalSlotsPerLeaf: logicalSlotsPerLeaf(topology) });
}

function executionRuntime(topology: SliceTopology): SparseCM12LogicalOwnerRuntime {
  return { brickActive: brick => topology.bricks[brick]?.active !== false,
    acceptedBrickResolution: brick => topology.bricks[brick]!.resolution,
    templateBrickCellRange: (brick, resolution) => {
      const source = topology.bricks[brick]!;
      if (source.resolution !== resolution) throw new Error("slice TEI rung disagrees with topology");
      return [topology.cellBaseByBrick.get(source.key) ?? 0,
        source.active === false ? 0 : resolution ** 2];
    } };
}

function packetSchedule(image: SparseCM12TransportExecutionImage,
  generation: number): Uint32Array {
  const result: number[] = [], layout = image.layout;
  const packetBase = layout.slotPacketBaseOffsets[0];
  for (let packet = 0; packet < layout.packetCapacity; packet += 1) {
    const at = packetBase + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
    if (image.words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.generation] === generation
      && image.words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.first]
        !== SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID) result.push(packet);
  }
  return Uint32Array.from(result);
}

function remapWorldDirectory(words: Uint32Array, layout: SparseCM12WorldDirectoryLayout,
  compactToStable: Uint32Array, generation: number): Uint32Array {
  const result = words.slice(), e = SPARSE_CM12_WORLD_DIRECTORY_ENTRY;
  for (let slot = 0; slot < layout.capacity; slot += 1) {
    const at = layout.entryBaseWords + slot * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS;
    if (result[at + e.state] !== 2) continue;
    const compact = result[at + e.leaf]!;
    result[at + e.leaf] = compactToStable[compact] ?? INVALID;
  }
  const records = Array.from(compactToStable, (_stable, compact) =>
    result.slice(layout.leafBaseWords + compact * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS,
      layout.leafBaseWords + (compact + 1) * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS));
  for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
    const at = layout.leafBaseWords + leaf * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
    result.fill(0, at, at + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS);
    result[at + SPARSE_CM12_WORLD_DIRECTORY_LEAF.generation] = INVALID;
  }
  compactToStable.forEach((stable, compact) => result.set(records[compact]!,
    layout.leafBaseWords + stable * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS));
  const highWater = Math.max(0, ...Array.from(compactToStable, stable => stable + 1));
  result[layout.baseWords + SPARSE_CM12_WORLD_DIRECTORY_HEADER.nextLeaf] = highWater;
  result[layout.baseWords + SPARSE_CM12_WORLD_DIRECTORY_HEADER.generation] = generation;
  return result;
}

function remapTransportExecutionImage(image: SparseCM12TransportExecutionImage,
  compactToStable: Uint32Array): SparseCM12TransportExecutionImage {
  if (compactToStable.every((stable, compact) => stable === compact)) return image;
  const { layout } = image, words = image.words.slice();
  for (const slot of [0, 1] as const) {
    const leafBase = layout.slotLeafBaseOffsets[slot];
    const packetBase = layout.slotPacketBaseOffsets[slot];
    const tileBase = layout.slotSpatialTileBaseOffsets[slot];
    const leafRecords = Array.from(compactToStable, (_stable, compact) =>
      words.slice(leafBase + compact * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
        leafBase + (compact + 1) * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS));
    const packetRecords = Array.from(compactToStable, (_stable, compact) =>
      words.slice(packetBase + compact * 64 * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
        packetBase + (compact + 1) * 64 * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS));
    const tileRecords = Array.from(compactToStable, (_stable, compact) =>
      words.slice(tileBase + compact * layout.spatialTilesPerLeaf
        * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
      tileBase + (compact + 1) * layout.spatialTilesPerLeaf
        * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS));
    words.fill(0, leafBase, leafBase
      + layout.leafCapacity * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS);
    words.fill(0, packetBase, packetBase
      + layout.packetCapacity * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS);
    words.fill(0, tileBase, tileBase
      + layout.spatialTileCapacity * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS);
    compactToStable.forEach((stable, compact) => {
      words.set(leafRecords[compact]!, leafBase
        + stable * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS);
      words.set(packetRecords[compact]!, packetBase
        + stable * 64 * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS);
      const tiles = tileRecords[compact]!;
      for (let tile = 0; tile < layout.spatialTilesPerLeaf; tile += 1) {
        const at = tile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
        const packet = tiles[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId]!;
        if (packet !== INVALID) tiles[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId]
          = stable * 64 + packet % 64;
      }
      words.set(tiles, tileBase + stable * layout.spatialTilesPerLeaf
        * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS);
    });
  }
  return Object.freeze({ layout, words });
}

function remapHotTopology(source: SparseCM12HotTopology,
  compactToStable: Uint32Array, leafCapacity: number): SparseCM12HotTopology {
  if (compactToStable.every((stable, compact) => stable === compact)
    && source.layout.logicalOwner.residentBrickCount === leafCapacity) return source;
  const words = source.words.slice(), old = source.layout, logicalOwner = Object.freeze({
    ...old.logicalOwner, residentBrickCount: leafCapacity,
  });
  words[SPARSE_CM12_LOGICAL_OWNER_HEADER.residentBrickCount] = leafCapacity;
  const recordBase = old.logicalOwner.recordBaseWords;
  for (let key = 0; key < old.logicalOwner.logicalBrickCount; key += 1) {
    const at = recordBase + key * SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS;
    const packed = words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.ownerAndSpan]!;
    if (packed === INVALID) continue;
    const compact = packed >>> 5;
    words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.ownerAndSpan]
      = ((compactToStable[compact]! << 5) | (packed & 31)) >>> 0;
  }
  for (let cell = 0; cell < old.cellCount; cell += 1) {
    const at = old.cellBaseWords + cell * 8 + SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution;
    const packed = words[at]!, compact = packed >>> 5;
    words[at] = ((compactToStable[compact]! << 5) | (packed & 31)) >>> 0;
  }
  for (let requirement = 0; requirement < old.requirementCount; requirement += 1) {
    const at = old.requirementBaseWords + requirement, packed = words[at]!, compact = packed >>> 5;
    words[at] = ((compactToStable[compact]! << 5) | (packed & 31)) >>> 0;
  }
  // HTP's own resident-count authority is its embedded LOD header. Cell and
  // requirement words above now use those same physical WDR slots.
  words[old.headerBaseWords + SPARSE_CM12_HOT_TOPOLOGY_HEADER.atlasGeneration]
    = source.layout.atlasGeneration;
  const layout = Object.freeze({ ...old, logicalOwner });
  return Object.freeze({ layout, words });
}

function createImage(topology: SliceTopology, leafCapacity: number,
  wdrLayout: SparseCM12WorldDirectoryLayout,
  teiLayout: SparseCM12TransportExecutionImageLayout): SliceRuntimeImage {
  requireStableLeafSlots(topology, leafCapacity);
  const atlas = productionAtlas(topology), grid = productionGrid(topology, atlas);
  const compactLeafToStable = Uint32Array.from(topology.bricks, brick => brick.id);
  const logical = createSparseCM12LogicalOwnerDirectory(atlas);
  const transportExecutionImage = remapTransportExecutionImage(
    createSparseCM12TransportExecutionImage(atlas, logical,
      executionRuntime(topology), { generation: topology.generation, layout: teiLayout }),
    compactLeafToStable);
  const worldDirectoryWords = remapWorldDirectory(
    createSparseCM12WorldDirectoryInitialWords(wdrLayout, atlas), wdrLayout,
    compactLeafToStable, topology.generation);
  const hotTopology = remapHotTopology(createSparseCM12HotTopology(grid),
    compactLeafToStable, leafCapacity);
  return Object.freeze({ topology, atlas, grid,
    worldDirectoryLayout: wdrLayout, worldDirectoryWords,
    transportExecutionImage, hotTopology,
    cellInvocationOrder: Uint32Array.from(topology.cells, cell => cell.id),
    stableCellOrder: Uint32Array.from(topology.cells, cell => cell.stableLeafId),
    rowInvocationOrder: Uint32Array.from(topology.rows, row => row.id),
    transportPacketSchedule: packetSchedule(transportExecutionImage, topology.generation),
    compactLeafToStable });
}

function baseReceipt(image: SliceRuntimeImage, acceptedSlot: 0 | 1): SliceRuntimeAuthorityReceipt {
  return { phase: "accepted", acceptedGeneration: image.topology.generation,
    candidateGeneration: 0, acceptedLeafCount: image.topology.bricks.length,
    candidateLeafCount: 0, acceptedCellCount: image.topology.cells.length,
    candidateCellCount: 0, acceptedRowCount: image.topology.rows.length,
    candidateRowCount: 0, acceptedPacketCount: image.transportPacketSchedule.length,
    candidatePacketCount: 0, acceptedSlot, candidateSlot: (acceptedSlot ^ 1) as 0 | 1,
    fault: 0, firstFaultId: INVALID };
}

/** WDR free-list storage order; the next production allocation pops the tail. */
export function sliceRuntimeFreeLeaves(image: SliceRuntimeImage): Uint32Array {
  const layout = image.worldDirectoryLayout, words = image.worldDirectoryWords;
  const count = words[layout.baseWords + SPARSE_CM12_WORLD_DIRECTORY_HEADER.freeCount]!;
  return words.slice(layout.baseWords + layout.freeListBaseWords,
    layout.baseWords + layout.freeListBaseWords + count);
}

function preserveWorldFreeState(candidate: SliceRuntimeImage,
  accepted: SliceRuntimeImage): SliceRuntimeImage {
  const free = Array.from(sliceRuntimeFreeLeaves(accepted));
  if (free.length === 0) return candidate;
  const occupied = new Set(candidate.topology.bricks.map(brick => brick.id));
  const retained = free.filter(leaf => !occupied.has(leaf));
  const layout = candidate.worldDirectoryLayout, h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
  const words = candidate.worldDirectoryWords.slice(), base = layout.baseWords;
  words.fill(0, base + layout.freeListBaseWords,
    base + layout.freeListBaseWords + layout.leafCapacity - layout.initialLeaves);
  words.set(retained, base + layout.freeListBaseWords);
  words[base + h.freeCount] = retained.length;
  words[base + h.nextLeaf] = Math.max(words[base + h.nextLeaf]!,
    accepted.worldDirectoryWords[base + h.nextLeaf]!);
  return Object.freeze({ ...candidate, worldDirectoryWords: words });
}

export function createSliceRuntimeAuthority(topology: SliceTopology,
  options: SliceRuntimeAuthorityOptions = {}): SliceRuntimeAuthority {
  const initialHighWater = Math.max(0, ...topology.bricks.map(brick => brick.id + 1));
  const leafCapacity = options.leafCapacity ?? initialHighWater;
  requireStableLeafSlots(topology, leafCapacity);
  const maximumSpan = Math.max(1, ...topology.bricks.map(brick => brick.spanBricks ?? 1));
  const wdrLayout = createSparseCM12WorldDirectoryLayout({ initialLeaves: initialHighWater,
    growthLeaves: leafCapacity - initialHighWater,
    maximumSpanLog: Math.log2(maximumSpan) });
  const teiLayout = transportLayout(topology, leafCapacity);
  const accepted = createImage(topology, leafCapacity, wdrLayout, teiLayout);
  return Object.freeze({ leafCapacity, accepted,
    transportExecutionWords: accepted.transportExecutionImage.words.slice(),
    acceptedSlot: 0, receipt: Object.freeze(baseReceipt(accepted, 0)) });
}

/** Compile candidate WDR/HTP images and the inactive production TEI2 slot. */
export function stageSliceRuntimeAuthority(authority: SliceRuntimeAuthority,
  candidateTopology: SliceTopology): SliceRuntimeAuthority {
  const accepted = authority.accepted;
  if (authority.candidate || authority.receipt.phase !== "accepted") {
    throw new Error("slice runtime authority already has a staged candidate");
  }
  if (candidateTopology.generation <= accepted.topology.generation) {
    throw new Error("slice runtime candidate generation must advance");
  }
  if (candidateTopology.dimensions.some((value, axis) =>
    value !== accepted.topology.dimensions[axis])) {
    throw new Error("slice runtime candidate changes the fixed world dimensions");
  }
  try {
    const candidate = preserveWorldFreeState(createImage(candidateTopology,
      authority.leafCapacity, accepted.worldDirectoryLayout,
      accepted.transportExecutionImage.layout), accepted);
    const candidateSlot = (authority.acceptedSlot ^ 1) as 0 | 1;
    const words = authority.transportExecutionWords.slice();
    const layout = candidate.transportExecutionImage.layout;
    words.set(candidate.transportExecutionImage.words.subarray(layout.slotBaseWords[0],
      layout.slotBaseWords[0] + layout.slotStrideWords), layout.slotBaseWords[candidateSlot]);
    const receipt: SliceRuntimeAuthorityReceipt = { phase: "candidate-ready",
      acceptedGeneration: accepted.topology.generation,
      candidateGeneration: candidateTopology.generation,
      acceptedLeafCount: accepted.topology.bricks.length,
      candidateLeafCount: candidateTopology.bricks.length,
      acceptedCellCount: accepted.topology.cells.length,
      candidateCellCount: candidateTopology.cells.length,
      acceptedRowCount: accepted.topology.rows.length,
      candidateRowCount: candidateTopology.rows.length,
      acceptedPacketCount: accepted.transportPacketSchedule.length,
      candidatePacketCount: candidate.transportPacketSchedule.length,
      acceptedSlot: authority.acceptedSlot, candidateSlot, fault: 0, firstFaultId: INVALID };
    return Object.freeze({ ...authority, candidate, transportExecutionWords: words,
      receipt: Object.freeze(receipt) });
  } catch (error) {
    const firstFaultId = candidateTopology.bricks.length > authority.leafCapacity
      ? authority.leafCapacity : INVALID;
    return Object.freeze({ ...authority,
      receipt: Object.freeze({ ...authority.receipt,
        candidateGeneration: candidateTopology.generation,
        fault: 1, firstFaultId }) });
  }
}

/** Publish all candidate banks together; a failed candidate never reaches here. */
export function commitSliceRuntimeAuthority(authority: SliceRuntimeAuthority): SliceRuntimeAuthority {
  if (!authority.candidate || authority.receipt.phase !== "candidate-ready"
    || authority.receipt.fault !== 0) throw new Error("slice runtime candidate is not committable");
  const acceptedSlot = authority.receipt.candidateSlot;
  const accepted = authority.candidate;
  return Object.freeze({ leafCapacity: authority.leafCapacity, accepted,
    transportExecutionWords: authority.transportExecutionWords.slice(), acceptedSlot,
    receipt: Object.freeze(baseReceipt(accepted, acceptedSlot)) });
}

/** Drop candidate images without changing accepted generations or TEI slot. */
export function cancelSliceRuntimeAuthority(authority: SliceRuntimeAuthority): SliceRuntimeAuthority {
  if (!authority.candidate) return authority;
  return Object.freeze({ leafCapacity: authority.leafCapacity, accepted: authority.accepted,
    transportExecutionWords: authority.transportExecutionWords.slice(),
    acceptedSlot: authority.acceptedSlot,
    receipt: Object.freeze(baseReceipt(authority.accepted, authority.acceptedSlot)) });
}

const signedOrder = (value: number): number => ((value | 0) ^ 0x8000_0000) >>> 0;

function recomputeWorldBounds(words: Uint32Array,
  layout: SparseCM12WorldDirectoryLayout): void {
  const h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
  const l = SPARSE_CM12_WORLD_DIRECTORY_LEAF;
  const base = layout.baseWords;
  const limit = Math.min(words[base + h.nextLeaf]!, layout.leafCapacity);
  const minimum = [0xffff_ffff, 0xffff_ffff, 0xffff_ffff];
  const maximum = [0, 0, 0];
  let found = false;
  for (let leaf = 0; leaf < limit; leaf += 1) {
    const at = base + layout.leafBaseWords
      + leaf * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
    if (words[at + l.generation] === INVALID) continue;
    const span = 2 ** words[at + l.spanLog]!;
    for (let axis = 0; axis < 3; axis += 1) {
      const q = words[at + axis]! | 0;
      minimum[axis] = Math.min(minimum[axis]!, signedOrder(q));
      maximum[axis] = Math.max(maximum[axis]!, signedOrder(q + span));
    }
    found = true;
  }
  if (!found) {
    minimum.fill(signedOrder(0));
    maximum.fill(signedOrder(0));
  }
  words[base + h.minimumX] = minimum[0]!;
  words[base + h.minimumY] = minimum[1]!;
  words[base + h.minimumZ] = minimum[2]!;
  words[base + h.maximumX] = maximum[0]!;
  words[base + h.maximumY] = maximum[1]!;
  words[base + h.maximumZ] = maximum[2]!;
  words[base + h.boundsGeneration] = words[base + h.generation]!;
}

/**
 * Execute production's post-presentation WDR release and frontier TEI scrub.
 * HTP remains the accepted-generation graph until the next topology commit;
 * only the mutable world directory and both stable TEI leaf ranges change.
 */
export function releaseSliceRuntimeLeaves(authority: SliceRuntimeAuthority,
  leafIds: readonly number[]): SliceRuntimeReleaseResult {
  if (authority.candidate || authority.receipt.phase !== "accepted") {
    throw new Error("slice runtime leaves cannot retire while a candidate is staged");
  }
  const requested = Uint32Array.from(leafIds, leaf => {
    if (!Number.isSafeInteger(leaf) || leaf < 0 || leaf >= authority.leafCapacity) {
      throw new RangeError(`slice runtime release leaf ${leaf} is outside the WDR arena`);
    }
    return leaf;
  });
  const image = authority.accepted;
  const layout = image.worldDirectoryLayout;
  const words = image.worldDirectoryWords.slice();
  const teiWords = authority.transportExecutionWords.slice();
  const h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
  const e = SPARSE_CM12_WORLD_DIRECTORY_ENTRY;
  const l = SPARSE_CM12_WORLD_DIRECTORY_LEAF;
  const base = layout.baseWords;
  const growthCapacity = layout.leafCapacity - layout.initialLeaves;
  const released: number[] = [], rejected: number[] = [];
  let fault = 0, firstFaultId = INVALID;
  for (const leaf of requested) {
    const leafAt = base + layout.leafBaseWords
      + leaf * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
    if (leaf < layout.initialLeaves || leaf >= words[base + h.nextLeaf]!
      || words[leafAt + l.generation] === INVALID) {
      rejected.push(leaf); continue;
    }
    const coordinate = [words[leafAt + l.x]! | 0, words[leafAt + l.y]! | 0,
      words[leafAt + l.z]! | 0] as const;
    const spanLog = words[leafAt + l.spanLog]!;
    const hash = sparseCM12WorldCoordinateHash(coordinate, spanLog);
    let slot = hash & (layout.capacity - 1), entryAt = -1;
    for (let probe = 0; probe < layout.capacity; probe += 1) {
      const at = base + layout.entryBaseWords
        + slot * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS;
      const state = words[at + e.state]!;
      if (state === 0) break;
      if (state === 2 && words[at + e.hash] === hash && words[at + e.leaf] === leaf) {
        entryAt = at; break;
      }
      slot = (slot + 1) & (layout.capacity - 1);
    }
    if (entryAt < 0) { rejected.push(leaf); continue; }
    const free = words[base + h.freeCount]!;
    if (free >= growthCapacity) {
      words[base + h.capacityFaults] = (words[base + h.capacityFaults]! + 1) >>> 0;
      fault = 1; if (firstFaultId === INVALID) firstFaultId = leaf;
      rejected.push(leaf); continue;
    }
    // cm12WorldReleaseLeaf invalidates only the descriptor generation. The
    // stale coordinate/span words remain deliberately unreadable until reuse.
    words[entryAt + e.state] = 3;
    words[leafAt + l.generation] = INVALID;
    words[base + h.liveCount] = (words[base + h.liveCount]! - 1) >>> 0;
    words[base + layout.freeListBaseWords + free] = leaf;
    words[base + h.freeCount] = free + 1;
    released.push(leaf);

    const tei = image.transportExecutionImage.layout;
    const generation = image.topology.generation;
    for (const bank of [0, 1] as const) {
      const slotAt = tei.slotBaseWords[bank];
      teiWords[slotAt] = generation;
      teiWords[slotAt + 1] = 1;
      const teiLeaf = tei.slotLeafBaseOffsets[bank]
        + leaf * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS;
      teiWords[teiLeaf] = generation;
      teiWords[teiLeaf + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF.flags]
        &= 0x7fff_ffff;
      for (let local = 0; local < tei.packetsPerLeaf; local += 1) {
        const packet = tei.slotPacketBaseOffsets[bank]
          + (leaf * tei.packetsPerLeaf + local)
            * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
        teiWords[packet + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.generation] = generation;
        teiWords[packet + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.first] = INVALID;
        teiWords[packet + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.counts] = 0;
        teiWords[packet + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.strides] = 0;
      }
      for (let local = 0; local < tei.spatialTilesPerLeaf; local += 1) {
        const tile = tei.slotSpatialTileBaseOffsets[bank]
          + (leaf * tei.spatialTilesPerLeaf + local)
            * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
        teiWords[tile + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.generation] = generation;
        teiWords[tile + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId] = INVALID;
        teiWords[tile + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskLow] = 0;
        teiWords[tile + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskHigh] = 0;
      }
    }
  }
  recomputeWorldBounds(words, layout);
  const transportExecutionImage = Object.freeze({
    layout: image.transportExecutionImage.layout, words: teiWords.slice(),
  });
  const releasedSet = new Set(released);
  const accepted = Object.freeze({ ...image, worldDirectoryWords: words,
    transportExecutionImage,
    transportPacketSchedule: Uint32Array.from(Array.from(image.transportPacketSchedule)
      .filter(packet => !releasedSet.has(Math.floor(packet / 64)))) });
  const nextAuthority = Object.freeze({ ...authority, accepted,
    transportExecutionWords: teiWords,
    receipt: Object.freeze(baseReceipt(accepted, authority.acceptedSlot)) });
  const receipt = Object.freeze({ requestedLeafIds: requested,
    releasedLeafIds: Uint32Array.from(released), rejectedLeafIds: Uint32Array.from(rejected),
    generation: words[base + h.generation]!, liveLeafCount: words[base + h.liveCount]!,
    freeLeafCount: words[base + h.freeCount]!, nextLeaf: words[base + h.nextLeaf]!,
    boundsGeneration: words[base + h.boundsGeneration]!, fault, firstFaultId });
  return Object.freeze({ authority: nextAuthority, receipt });
}

/** WDR next-leaf receipt, kept as an accessor to the exact production header. */
export function sliceRuntimeNextLeaf(image: SliceRuntimeImage): number {
  return image.worldDirectoryWords[image.worldDirectoryLayout.baseWords
    + SPARSE_CM12_WORLD_DIRECTORY_HEADER.nextLeaf]!;
}
