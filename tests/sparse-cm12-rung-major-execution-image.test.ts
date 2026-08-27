import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createSparseCM12LogicalOwnerDirectory } from
  "../lib/methods/adaptive-mass/sparse-cm12-logical-owner-directory";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  createSparseCM12TransportExecutionImage,
  createSparseCM12TransportExecutionImageLayout,
  decodeSparseCM12TransportExecutionImageLeafScaleLog2,
  encodeSparseCM12TransportExecutionImageLeafScaleDescriptor,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
} from "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image";
import { createSparseCM12TransportExecutionImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.wgsl";

const residentHostSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const residentWGSLSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

const popcount = (word: number) => {
  let value = word >>> 0;
  let count = 0;
  while (value !== 0) {
    value = (value & (value - 1)) >>> 0;
    count += 1;
  }
  return count;
};

function fixture(options: {
  dimensions?: SparseBrickVec3;
  resolution: SparseBrickResolution;
  spanBricks?: number;
  active?: boolean;
  leafCapacity?: number;
}) {
  const dimensions = options.dimensions ?? [16, 16, 16];
  const brickFineResolution = 16 as const;
  const brickDimensions = dimensions.map((value) =>
    Math.ceil(value / brickFineResolution)) as [number, number, number];
  const coordinate = [0, 0, 0] as const;
  const count = options.resolution ** 3;
  const atlas = createSparseAdaptiveMassAtlas(dimensions, [{
    key: sparseBrickKey(coordinate, brickDimensions), coordinate,
    spanBricks: options.spanBricks,
    resolution: options.resolution,
    density: new Float64Array(count), gamma: new Float64Array(count),
  }], 7, brickFineResolution);
  const directory = createSparseCM12LogicalOwnerDirectory(atlas);
  const scale = brickFineResolution * (options.spanBricks ?? 1) / options.resolution;
  const valid = dimensions.map((value) => Math.ceil(value / scale));
  const liveCount = valid[0]! * valid[1]! * valid[2]!;
  const span = options.spanBricks ?? 1;
  const extent = brickDimensions.map((value) => Math.min(span, value));
  const logicalSlotsPerLeaf = (extent[2]! - 1) * span * span
    + (extent[1]! - 1) * span + extent[0]!;
  const layout = options.leafCapacity === undefined ? undefined
    : createSparseCM12TransportExecutionImageLayout({
      brickFineResolution,
      logicalBrickDimensions: directory.layout.logicalBrickDimensions,
      leafCapacity: options.leafCapacity,
      maximumSpanBricks: atlas.maximumSpanBricks,
      logicalSlotsPerLeaf,
    });
  const image = createSparseCM12TransportExecutionImage(atlas, directory, {
    brickActive: () => options.active ?? true,
    acceptedBrickResolution: () => options.resolution,
    templateBrickCellRange: () => [1000, liveCount],
  }, { generation: 19, layout });
  return { image, valid, liveCount, scale };
}

test("TEI2 reserves the resident growth capacity without inventing host leaves", () => {
  const { image } = fixture({ resolution: 16, leafCapacity: 5 });
  assert.equal(image.layout.leafCapacity, 5);
  assert.equal(image.words.length, image.layout.totalWords);
  for (const leafBase of image.layout.slotLeafBaseOffsets) {
    assert.equal(image.words[leafBase], 19);
    assert.equal(image.words[leafBase + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS],
      0);
  }
});

test("TEI2 scale descriptors extend word 7 without changing the leaf ABI", () => {
  assert.equal(SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS, 8);
  for (let scaleLog2 = 0; scaleLog2 <= 31; scaleLog2 += 1) {
    const scale = 2 ** scaleLog2;
    for (const packetAxis of [1, 2, 4, 255]) {
      const descriptor = encodeSparseCM12TransportExecutionImageLeafScaleDescriptor(
        scale, packetAxis,
      );
      assert.equal(descriptor
        & SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR.packetAxisMask,
      packetAxis);
      assert.equal(decodeSparseCM12TransportExecutionImageLeafScaleLog2(
        scale, descriptor,
      ), scaleLog2);
      // Original TEI2 images carried only packetAxis in word 7.
      assert.equal(decodeSparseCM12TransportExecutionImageLeafScaleLog2(
        scale, packetAxis,
      ), scaleLog2);
    }
  }
  assert.throws(() => encodeSparseCM12TransportExecutionImageLeafScaleDescriptor(3, 1),
    /power of two/);
  assert.throws(() => decodeSparseCM12TransportExecutionImageLeafScaleLog2(
    4, encodeSparseCM12TransportExecutionImageLeafScaleDescriptor(8, 1),
  ), /disagrees/);
});

test("TEI2 dyadic shift/mask owner arithmetic is exhaustive over supported scales", () => {
  const scales = new Set<number>();
  for (const brickFine of [4, 8, 16]) {
    for (const span of [1, 2, 4, 8, 16]) {
      for (const resolution of [1, 2, 4, 8, 16]) {
        if (resolution <= brickFine) scales.add(brickFine * span / resolution);
      }
    }
  }
  for (const scale of [...scales].sort((a, b) => a - b)) {
    const scaleLog2 = Math.log2(scale);
    const origin = 3 * scale;
    // Division and multiplication in the CPU reference are separable by axis.
    // Cover every residue in four consecutive cells for every realizable scale.
    for (let value = origin; value < origin + 4 * scale; value += 1) {
      const relative = value - origin;
      const divisionLocal = Math.floor(relative / scale);
      const shiftedLocal = relative >>> scaleLog2;
      assert.equal(shiftedLocal, divisionLocal, `scale ${scale}, fine ${value}`);
      const multipliedLower = origin + divisionLocal * scale;
      const maskedLower = origin + (relative & ~(scale - 1));
      assert.equal(maskedLower, multipliedLower, `scale ${scale}, fine ${value}`);
    }

    const valid = [4, 3, 2] as const;
    const dimensions = [origin + 4 * scale - Math.min(scale - 1, 1),
      origin + 3 * scale - Math.min(scale - 1, 2),
      origin + 2 * scale - Math.min(scale - 1, 3)] as const;
    for (let z = 0; z < valid[2]; z += 1) {
      for (let y = 0; y < valid[1]; y += 1) {
        for (let x = 0; x < valid[0]; x += 1) {
          const divisionOffset = x + valid[0] * (y + valid[1] * z);
          for (const rx of [0, scale - 1]) {
            for (const ry of [0, scale - 1]) {
              for (const rz of [0, scale - 1]) {
                const q = [origin + x * scale + rx, origin + y * scale + ry,
                  origin + z * scale + rz] as const;
                if (q.some((value, axis) => value >= dimensions[axis]!)) continue;
                const shifted = q.map((value) => (value - origin) >>> scaleLog2);
                const shiftedOffset = shifted[0]!
                  + valid[0] * (shifted[1]! + valid[1] * shifted[2]!);
                assert.equal(shiftedOffset, divisionOffset);
                const divisionLower = q.map((value) => origin
                  + Math.floor((value - origin) / scale) * scale);
                const maskedLower = q.map((value) => origin
                  + ((value - origin) & ~(scale - 1)));
                assert.deepEqual(maskedLower, divisionLower);
                assert.deepEqual(maskedLower.map((lower, axis) =>
                  Math.min(scale, Math.max(0, dimensions[axis]! - lower))),
                divisionLower.map((lower, axis) =>
                  Math.min(scale, Math.max(0, dimensions[axis]! - lower))));
              }
            }
          }
        }
      }
    }
  }
});

test("TEI2 WGSL hot owner path preserves signed world coordinates", () => {
  const source = createSparseCM12TransportExecutionImageWGSL({
    layout: createSparseCM12TransportExecutionImageLayout({
      brickFineResolution: 16, logicalBrickDimensions: [2, 2, 2], leafCapacity: 4,
    }),
  });
  const owner = source.slice(source.indexOf("fn cm12TeiOwnerAtFine"),
    source.indexOf("fn cm12TeiPacket(", source.indexOf("fn cm12TeiOwnerAtFine")));
  assert.match(owner, /let leaf=cm12TeiLeafAtLogical\(logical\);let owner=leaf\.owner;/,
    "the hot fine owner resolver must consume the staged WDR owner");
  assert.doesNotMatch(owner, /cm12WorldOwnerAt/,
    "the hot fine owner resolver must not bypass the staged directory");
  assert.match(owner, /let scale=1u<<leaf\.scaleLog2/);
  assert.match(owner, /let relative=q-origin/);
  assert.match(owner, /if\(any\(relative<vec3i\(0\)\)\)/);
  assert.match(owner, /let shift=vec3u\(leaf\.scaleLog2\)/);
  assert.match(owner, /let local=vec3u\(relative\)>>shift/);
  assert.match(owner, /let lower=origin\+vec3i\(vec3u\(relative\)&~vec3u\(scale-1u\)\)/);
  assert.doesNotMatch(owner, /\/leaf\.scale|\*leaf\.scale/,
    "hot owner resolution must not restore runtime scale division/multiplication");
  assert.match(source, /leaf\.scale,leaf\.scaleLog2/,
    "the 27-leaf cache must carry the decoded shift");
  assert.match(source, /cm12TeiCache1\[lane\]=vec4u\(leaf\.owner,/,
    "the 27-leaf cache must carry signed-world physical ownership");
  assert.match(source,
    /\[at\+7u\]=CM12_TEI_SCALE_LOG2_ENCODED[\s\S]*firstLeadingBit\(scale\)/,
    "the topology compiler must publish the backwards-compatible descriptor");
});

for (const [resolution, expectedPackets] of [
  [1, 1], [2, 1], [4, 1], [8, 8], [16, 64],
] as const) {
  test(`TEI2 B16 rung ${resolution} publishes ${expectedPackets} dense home packets`, () => {
    const { image, liveCount, scale } = fixture({ resolution });
    const { layout, words } = image;
    assert.equal(layout.packetCapacity, 64);
    assert.equal(layout.spatialTileCapacity, 64);
    for (const slot of [0, 1] as const) {
      const leaf = layout.slotLeafBaseOffsets[slot];
      const descriptor = words[leaf
        + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF.scaleDescriptor]!;
      assert.notEqual(descriptor
        & SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR.encodedMask, 0);
      assert.equal(decodeSparseCM12TransportExecutionImageLeafScaleLog2(
        words[leaf + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF.scale]!, descriptor,
      ), Math.log2(scale));
      const packets = layout.slotPacketBaseOffsets[slot];
      const active: number[] = [];
      for (let packet = 0; packet < layout.packetCapacity; packet += 1) {
        const at = packets + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
        if ((words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.counts]!
            & 0x8000_0000) !== 0) active.push(packet);
      }
      assert.deepEqual(active, Array.from({ length: expectedPackets }, (_, id) => id));

      let selectedCells = 0;
      const tiles = layout.slotSpatialTileBaseOffsets[slot];
      for (let tile = 0; tile < layout.spatialTileCapacity; tile += 1) {
        const at = tiles + tile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
        const packet = words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId]!;
        assert.notEqual(packet, SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID);
        assert.ok(packet < expectedPackets);
        selectedCells += popcount(words[
          at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskLow]!)
          + popcount(words[
            at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskHigh]!);
      }
      assert.equal(selectedCells, liveCount,
        "each accepted cell lower corner occurs in exactly one spatial-tile mask");
    }
  });
}

test("TEI2 clips packet counts and owner-lower masks at the world edge", () => {
  const { image, valid, liveCount } = fixture({
    dimensions: [13, 10, 7], resolution: 16,
  });
  assert.deepEqual(valid, [13, 10, 7]);
  const { layout, words } = image;
  let packetCells = 0;
  let activePackets = 0;
  for (let packet = 0; packet < 64; packet += 1) {
    const at = layout.slotPacketBaseOffsets[0]
      + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
    const packed = words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.counts]!;
    if ((packed & 0x8000_0000) === 0) continue;
    activePackets += 1;
    packetCells += (packed & 31) * ((packed >>> 5) & 31) * ((packed >>> 10) & 31);
  }
  assert.equal(activePackets, Math.ceil(valid[0]! / 4)
    * Math.ceil(valid[1]! / 4) * Math.ceil(valid[2]! / 4));
  assert.equal(packetCells, liveCount);
});

test("TEI2 maps logical tiles across a clipped macro leaf into its home packets", () => {
  const { image, liveCount } = fixture({
    dimensions: [32, 16, 16], resolution: 8, spanBricks: 2,
  });
  const { layout, words } = image;
  assert.equal(layout.leafCapacity, 1);
  assert.equal(layout.packetCapacity, 64);
  assert.equal(layout.spatialTileCapacity, 128);
  let selectedCells = 0;
  const referenced = new Set<number>();
  for (let tile = 0; tile < layout.spatialTileCapacity; tile += 1) {
    const at = layout.slotSpatialTileBaseOffsets[0]
      + tile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
    referenced.add(words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId]!);
    selectedCells += popcount(words[
      at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskLow]!)
      + popcount(words[
        at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskHigh]!);
  }
  assert.deepEqual([...referenced].sort((a, b) => a - b), [0, 1]);
  assert.equal(selectedCells, liveCount);
});

test("TEI2 inactive leaves publish no packets or spatial authority", () => {
  const { image } = fixture({ resolution: 8, active: false });
  const { layout, words } = image;
  for (const slot of [0, 1] as const) {
    for (let packet = 0; packet < layout.packetCapacity; packet += 1) {
      const at = layout.slotPacketBaseOffsets[slot]
        + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
      assert.equal(words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.first],
        SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID);
    }
    for (let tile = 0; tile < layout.spatialTileCapacity; tile += 1) {
      const at = layout.slotSpatialTileBaseOffsets[slot]
        + tile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
      assert.equal(words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.packetId],
        SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID);
    }
  }
});
