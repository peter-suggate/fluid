import {
  sparseCM12LogicalOwnerAtKey,
  type SparseCM12LogicalOwnerDirectory,
  type SparseCM12LogicalOwnerRuntime,
} from "./sparse-cm12-logical-owner-directory";
import { sparseBrickSpan, type SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";

/** Accepted transport execution image, version 2 (rung-major packets). */
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_MAGIC = 0x5445_4932; // TEI2
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_VERSION = 2;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_HEADER_WORDS = 24;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SLOT_HEADER_WORDS = 8;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS = 8;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS = 4;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS = 4;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF = 64;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE = 4;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_AXIS_CAPACITY = 4;
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID = 0xffff_ffff;

export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF = Object.freeze({
  generation: 0, flags: 1, cellFirst: 2, cellCount: 3,
  originKey: 4, validDimensions: 5, scale: 6, scaleDescriptor: 7, reserved: 7,
} as const);
/**
 * Backwards-compatible use of TEI2 leaf word 7.  The low byte retains the
 * packet-axis value published by the original TEI2 producer.  New producers
 * also publish log2(scale), guarded by a marker that old consumers ignore.
 * New consumers fall back to deriving the shift from word 6 when reading an
 * image written by the original producer.
 */
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR =
  Object.freeze({
    packetAxisMask: 0xff,
    scaleLog2Shift: 8,
    scaleLog2Mask: 0x1f,
    encodedMask: 0x8000_0000,
  } as const);
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET = Object.freeze({
  generation: 0, first: 1, counts: 2, strides: 3,
} as const);
/** A stable finest-lattice 4^3 tile maps to one home-rung packet and its lanes. */
export const SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE = Object.freeze({
  generation: 0, packetId: 1, laneMaskLow: 2, laneMaskHigh: 3,
} as const);

export interface SparseCM12TransportExecutionImageLayout {
  readonly brickFineResolution: 4 | 8 | 16;
  readonly logicalBrickDimensions: readonly [number, number, number];
  readonly leafCapacity: number;
  /** Stable ABI: packetId = leaf * 64 + rung-local packet ordinal. */
  readonly packetsPerLeaf: 64;
  readonly packetCapacity: number;
  readonly packetEdge: 4;
  readonly packetAxisCapacity: 4;
  readonly spatialTilesPerLogicalBrickAxis: number;
  readonly spatialTilesPerLogicalBrick: number;
  readonly spatialTileCapacity: number;
  /** Compatibility aliases for the stable spatial-dirty-tile lattice. */
  readonly tilesPerLogicalBrickAxis: number;
  readonly tilesPerLogicalBrick: number;
  readonly slotBaseWords: readonly [number, number];
  readonly slotStrideWords: number;
  readonly slotLeafBaseOffsets: readonly [number, number];
  readonly slotPacketBaseOffsets: readonly [number, number];
  readonly slotSpatialTileBaseOffsets: readonly [number, number];
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12TransportExecutionImage {
  readonly layout: SparseCM12TransportExecutionImageLayout;
  readonly words: Uint32Array;
}

const align = (value: number, alignment: number) =>
  Math.ceil(value / alignment) * alignment;
const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};

const checkedDyadicScaleLog2 = (scale: number): number => {
  const scaleLog2 = Math.log2(scale);
  if (!Number.isSafeInteger(scale) || scale <= 0 || scale > 0x8000_0000
    || !Number.isInteger(scaleLog2)) {
    throw new RangeError("TEI2 leaf scale must be a positive u32 power of two");
  }
  return scaleLog2;
};

export function encodeSparseCM12TransportExecutionImageLeafScaleDescriptor(
  scale: number,
  packetAxis: number,
): number {
  const d = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR;
  if (!Number.isSafeInteger(packetAxis) || packetAxis <= 0
    || packetAxis > d.packetAxisMask) {
    throw new RangeError("TEI2 leaf packet axis is outside the descriptor range");
  }
  return (d.encodedMask | packetAxis
    | (checkedDyadicScaleLog2(scale) << d.scaleLog2Shift)) >>> 0;
}

/** Decode both current descriptors and original TEI2 packet-axis-only word 7. */
export function decodeSparseCM12TransportExecutionImageLeafScaleLog2(
  scale: number,
  descriptor: number,
): number {
  const d = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR;
  const fallback = checkedDyadicScaleLog2(scale);
  if (((descriptor >>> 0) & d.encodedMask) === 0) return fallback;
  const encoded = ((descriptor >>> d.scaleLog2Shift) & d.scaleLog2Mask) >>> 0;
  if (2 ** encoded !== scale) {
    throw new Error("TEI2 leaf scale descriptor disagrees with its scale word");
  }
  return encoded;
}

export function createSparseCM12TransportExecutionImageLayout(options: {
  readonly brickFineResolution: 4 | 8 | 16;
  readonly logicalBrickDimensions: readonly [number, number, number];
  readonly leafCapacity: number;
}): SparseCM12TransportExecutionImageLayout {
  const spatialTilesPerLogicalBrickAxis = options.brickFineResolution
    / SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE;
  const spatialTilesPerLogicalBrick = spatialTilesPerLogicalBrickAxis ** 3;
  const logicalCount = checked(options.logicalBrickDimensions.reduce((a, b) =>
    checked(a * checked(b, "logicalBrickDimension"), "logicalBrickCount"), 1),
  "logicalBrickCount");
  const spatialTileCapacity = checked(logicalCount * spatialTilesPerLogicalBrick,
    "spatialTileCapacity");
  const leafCapacity = checked(options.leafCapacity, "leafCapacity");
  const packetCapacity = checked(leafCapacity
    * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF, "packetCapacity");
  const leafWords = checked(leafCapacity
    * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS, "leafWords");
  const packetWords = checked(packetCapacity
    * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS, "packetWords");
  const spatialTileWords = checked(spatialTileCapacity
    * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS, "spatialTileWords");
  const slotStrideWords = align(
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SLOT_HEADER_WORDS
      + leafWords + packetWords + spatialTileWords,
    64,
  );
  const slot0 = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_HEADER_WORDS;
  const slot1 = checked(slot0 + slotStrideWords, "slot1BaseWords");
  const leaf0 = slot0 + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SLOT_HEADER_WORDS;
  const leaf1 = slot1 + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SLOT_HEADER_WORDS;
  const packet0 = leaf0 + leafWords;
  const packet1 = leaf1 + leafWords;
  const spatialTile0 = packet0 + packetWords;
  const spatialTile1 = packet1 + packetWords;
  const totalWords = checked(slot1 + slotStrideWords, "totalWords");
  return Object.freeze({
    brickFineResolution: options.brickFineResolution,
    logicalBrickDimensions: [...options.logicalBrickDimensions] as [number, number, number],
    leafCapacity,
    packetsPerLeaf: SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF,
    packetCapacity,
    packetEdge: SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE,
    packetAxisCapacity: SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_AXIS_CAPACITY,
    spatialTilesPerLogicalBrickAxis, spatialTilesPerLogicalBrick, spatialTileCapacity,
    tilesPerLogicalBrickAxis: spatialTilesPerLogicalBrickAxis,
    tilesPerLogicalBrick: spatialTilesPerLogicalBrick,
    slotBaseWords: [slot0, slot1] as const, slotStrideWords,
    slotLeafBaseOffsets: [leaf0, leaf1] as const,
    slotPacketBaseOffsets: [packet0, packet1] as const,
    slotSpatialTileBaseOffsets: [spatialTile0, spatialTile1] as const,
    totalWords, totalBytes: 4 * totalWords,
  });
}

const keyCoordinate = (key: number, dimensions: readonly number[]) => {
  const xy = dimensions[0]! * dimensions[1]!;
  const z = Math.floor(key / xy);
  const remainder = key - z * xy;
  const y = Math.floor(remainder / dimensions[0]!);
  return [remainder - y * dimensions[0]!, y, z] as const;
};
const pack3x5 = (value: readonly number[]) =>
  (value[0]! | (value[1]! << 5) | (value[2]! << 10)) >>> 0;
const packetAxisForResolution = (resolution: number) => Math.max(1,
  Math.ceil(resolution / SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE));
const packetLocalId = (coordinate: readonly number[], axis: number) =>
  coordinate[0]! + axis * (coordinate[1]! + axis * coordinate[2]!);

function writeSlot(
  words: Uint32Array,
  layout: SparseCM12TransportExecutionImageLayout,
  slot: 0 | 1,
  atlas: SparseAdaptiveMassAtlas,
  directory: SparseCM12LogicalOwnerDirectory,
  runtime: SparseCM12LogicalOwnerRuntime,
  generation: number,
): void {
  const slotBase = layout.slotBaseWords[slot];
  const leafBase = layout.slotLeafBaseOffsets[slot];
  const packetBase = layout.slotPacketBaseOffsets[slot];
  const spatialTileBase = layout.slotSpatialTileBaseOffsets[slot];
  words.set([
    generation, 1, layout.leafCapacity, layout.packetCapacity,
    layout.spatialTileCapacity, 0, 0, 0,
  ], slotBase);

  for (let brick = 0; brick < layout.leafCapacity; brick += 1) {
    const source = atlas.bricks[brick]!;
    const resolution = runtime.acceptedBrickResolution(brick);
    const active = runtime.brickActive(brick);
    const range = runtime.templateBrickCellRange(brick, resolution);
    const spanBricks = sparseBrickSpan(source);
    const spanFine = layout.brickFineResolution * spanBricks;
    if (resolution > layout.brickFineResolution || spanFine % resolution !== 0) {
      throw new Error(`TEI2 leaf ${brick} has invalid accepted rung ${resolution}`);
    }
    const scale = spanFine / resolution;
    const originFine = source.coordinate.map((v) => v * layout.brickFineResolution);
    const extent = originFine.map((v, axis) => Math.max(0, Math.min(
      spanFine, atlas.dimensions[axis]! - v,
    )));
    const valid = extent.map((v) => Math.ceil(v / scale));
    const at = leafBase + brick * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS;
    words[at] = generation;
    words[at + 1] = (resolution | (Math.log2(spanBricks) << 8)
      | (active ? 0x8000_0000 : 0)) >>> 0;
    words[at + 2] = range[0];
    words[at + 3] = range[1];
    words[at + 4] = source.key;
    words[at + 5] = pack3x5(valid);
    words[at + 6] = scale;
    words[at + 7] = encodeSparseCM12TransportExecutionImageLeafScaleDescriptor(
      scale, packetAxisForResolution(resolution),
    );

    const packetAxis = packetAxisForResolution(resolution);
    for (let localPacket = 0;
      localPacket < SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF;
      localPacket += 1) {
      const packetId = brick * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF
        + localPacket;
      const packetAt = packetBase
        + packetId * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
      words[packetAt] = generation;
      words[packetAt + 1] = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID;
      words[packetAt + 2] = 0;
      words[packetAt + 3] = 0;
      if (!active || localPacket >= packetAxis ** 3) continue;
      const pz = Math.floor(localPacket / (packetAxis ** 2));
      const remainder = localPacket - pz * packetAxis ** 2;
      const py = Math.floor(remainder / packetAxis);
      const px = remainder - py * packetAxis;
      const local = [4 * px, 4 * py, 4 * pz] as const;
      const counts = local.map((v, axis) => Math.max(0,
        Math.min(4, valid[axis]! - v)));
      if (counts.some((v) => v === 0)) continue;
      const first = range[0] + local[0] + valid[0]!
        * (local[1] + valid[1]! * local[2]);
      if (first >= range[0] + range[1]) continue;
      words[packetAt + 1] = first;
      words[packetAt + 2] = (pack3x5(counts) | 0x8000_0000) >>> 0;
      words[packetAt + 3] = (valid[0]!
        | ((valid[0]! * valid[1]!) << 16)) >>> 0;
    }
  }

  const dims = layout.logicalBrickDimensions;
  for (let spatialTile = 0; spatialTile < layout.spatialTileCapacity; spatialTile += 1) {
    const logicalKey = Math.floor(spatialTile / layout.spatialTilesPerLogicalBrick);
    const tile = spatialTile % layout.spatialTilesPerLogicalBrick;
    const logical = keyCoordinate(logicalKey, dims);
    const tx = tile % layout.spatialTilesPerLogicalBrickAxis;
    const ty = Math.floor(tile / layout.spatialTilesPerLogicalBrickAxis)
      % layout.spatialTilesPerLogicalBrickAxis;
    const tz = Math.floor(tile / (layout.spatialTilesPerLogicalBrickAxis ** 2));
    const origin = [
      logical[0] * layout.brickFineResolution + 4 * tx,
      logical[1] * layout.brickFineResolution + 4 * ty,
      logical[2] * layout.brickFineResolution + 4 * tz,
    ] as const;
    const at = spatialTileBase
      + spatialTile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
    words[at] = generation;
    words[at + 1] = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID;
    words[at + 2] = 0;
    words[at + 3] = 0;
    const owner = sparseCM12LogicalOwnerAtKey(directory, logicalKey);
    if (!owner || !runtime.brickActive(owner.brick)
      || origin.some((v, axis) => v >= atlas.dimensions[axis]!)) continue;
    const resolution = runtime.acceptedBrickResolution(owner.brick);
    const spanFine = layout.brickFineResolution * owner.spanBricks;
    const scale = spanFine / resolution;
    const leafOrigin = owner.origin.map((v) => v * layout.brickFineResolution);
    const extent = leafOrigin.map((v, axis) => Math.max(0, Math.min(
      spanFine, atlas.dimensions[axis]! - v,
    )));
    const valid = extent.map((v) => Math.ceil(v / scale));
    const home = origin.map((v, axis) => Math.floor((v - leafOrigin[axis]!) / scale));
    const packetAxis = packetAxisForResolution(resolution);
    const packetCoordinate = home.map((v) => Math.floor(v / 4));
    if (packetCoordinate.some((v) => v < 0 || v >= packetAxis)) continue;
    const packetId = owner.brick
      * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF
      + packetLocalId(packetCoordinate, packetAxis);
    words[at + 1] = packetId;

    let low = 0;
    let high = 0;
    for (let lane = 0; lane < 64; lane += 1) {
      const q = [origin[0] + (lane & 3), origin[1] + ((lane >>> 2) & 3),
        origin[2] + (lane >>> 4)] as const;
      if (q.some((v, axis) => v >= atlas.dimensions[axis]!)) continue;
      const relative = q.map((v, axis) => v - leafOrigin[axis]!);
      if (relative.some((v) => v < 0 || v % scale !== 0)) continue;
      const local = relative.map((v) => v / scale);
      if (local.some((v, axis) => v >= valid[axis]!)) continue;
      const cellPacket = local.map((v) => Math.floor(v / 4));
      if (packetLocalId(cellPacket, packetAxis)
          !== packetId % SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF) {
        throw new Error(`TEI2 spatial tile ${spatialTile} crosses a home-rung packet`);
      }
      const cellLane = (local[0]! & 3) + 4 * ((local[1]! & 3) + 4 * (local[2]! & 3));
      if (cellLane < 32) low = (low | (1 << cellLane)) >>> 0;
      else high = (high | (1 << (cellLane - 32))) >>> 0;
    }
    words[at + 2] = low;
    words[at + 3] = high;
  }
}

export function createSparseCM12TransportExecutionImage(
  atlas: SparseAdaptiveMassAtlas,
  directory: SparseCM12LogicalOwnerDirectory,
  runtime: SparseCM12LogicalOwnerRuntime,
  options: { readonly generation?: number } = {},
): SparseCM12TransportExecutionImage {
  const layout = createSparseCM12TransportExecutionImageLayout({
    brickFineResolution: atlas.brickFineResolution,
    logicalBrickDimensions: directory.layout.logicalBrickDimensions,
    leafCapacity: atlas.bricks.length,
  });
  const generation = checked(options.generation ?? 1, "generation");
  const words = new Uint32Array(layout.totalWords);
  words.set([
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_MAGIC,
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_VERSION,
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_HEADER_WORDS,
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
    SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
    layout.brickFineResolution,
    layout.logicalBrickDimensions[0], layout.logicalBrickDimensions[1],
    layout.logicalBrickDimensions[2], layout.leafCapacity, layout.packetCapacity,
    layout.spatialTileCapacity, layout.spatialTilesPerLogicalBrickAxis,
    layout.packetsPerLeaf, layout.packetEdge,
    layout.slotBaseWords[0], layout.slotBaseWords[1], layout.slotStrideWords,
    layout.slotLeafBaseOffsets[0], layout.slotPacketBaseOffsets[0],
    layout.slotSpatialTileBaseOffsets[0], layout.totalWords, 0,
  ]);
  writeSlot(words, layout, 0, atlas, directory, runtime, generation);
  writeSlot(words, layout, 1, atlas, directory, runtime, generation);
  return Object.freeze({ layout, words });
}
