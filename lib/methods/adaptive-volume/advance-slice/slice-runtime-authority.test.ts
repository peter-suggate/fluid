import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_HEADER,
  SPARSE_CM12_HOT_TOPOLOGY_MAGIC,
} from "../sparse-cm12-hot-topology";
import {
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
} from "../sparse-cm12-transport-execution-image";
import {
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY,
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS,
  SPARSE_CM12_WORLD_DIRECTORY_HEADER,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS,
  SPARSE_CM12_WORLD_DIRECTORY_MAGIC,
} from "../sparse-cm12-world-directory";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import {
  cancelSliceRuntimeAuthority,
  commitSliceRuntimeAuthority,
  createSliceRuntimeAuthority,
  releaseSliceRuntimeLeaves,
  sliceRuntimeNextLeaf,
  stageSliceRuntimeAuthority,
} from "./slice-runtime-authority";
import { createAdvanceSlice } from "./slice-solver";
import { compileSliceTopology } from "./slice-topology";

const floatBits = (value: number): number =>
  new Uint32Array(new Float32Array([value]).buffer)[0]!;

test("the slice authority uses literal production WDR1, TEI2 and HTP1 banks", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const topology = slice.topology.accepted;
  const authority = createSliceRuntimeAuthority(topology,
    { leafCapacity: topology.bricks.length + 8 });
  const image = authority.accepted;

  assert.equal(image.worldDirectoryWords[SPARSE_CM12_WORLD_DIRECTORY_HEADER.magic],
    SPARSE_CM12_WORLD_DIRECTORY_MAGIC);
  assert.equal(sliceRuntimeNextLeaf(image), topology.bricks.length);
  const htpHeader = image.hotTopology.layout.headerBaseWords;
  assert.equal(image.hotTopology.words[htpHeader + SPARSE_CM12_HOT_TOPOLOGY_HEADER.magic],
    SPARSE_CM12_HOT_TOPOLOGY_MAGIC);
  assert.equal(image.hotTopology.layout.cellCount, topology.cells.length);
  assert.equal(image.hotTopology.layout.rowCount, topology.rows.length);
  assert.equal(image.hotTopology.layout.incidenceCount, topology.incidences.length);

  for (const cell of topology.cells) {
    const at = image.hotTopology.layout.cellBaseWords + 8 * cell.id;
    assert.equal(image.hotTopology.words[at + SPARSE_CM12_HOT_TOPOLOGY_CELL.centerZ],
      floatBits(0.5));
    assert.equal(image.hotTopology.words[at + SPARSE_CM12_HOT_TOPOLOGY_CELL.widthZ],
      floatBits(1));
    assert.equal(image.hotTopology.words[at + SPARSE_CM12_HOT_TOPOLOGY_CELL.volume],
      floatBits(cell.volumeFineCells));
  }
  assert.deepEqual(Array.from(image.stableCellOrder),
    topology.cells.map(cell => cell.stableLeafId));
  assert.deepEqual(Array.from(image.rowInvocationOrder),
    topology.rows.map(row => row.id));
});

test("TEI retains leaf-times-64 packet ids while only the z=0 lane plane is live", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const authority = createSliceRuntimeAuthority(slice.topology.accepted);
  const image = authority.accepted.transportExecutionImage;
  const layout = image.layout, words = image.words;
  for (const packet of authority.accepted.transportPacketSchedule) {
    assert.equal(Math.floor(packet / 64) < slice.topology.accepted.bricks.length, true);
    const at = layout.slotPacketBaseOffsets[0]
      + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
    assert.equal(words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.generation],
      slice.topology.accepted.generation);
  }
  for (let tile = 0; tile < layout.spatialTileCapacity; tile += 1) {
    const at = layout.slotSpatialTileBaseOffsets[0]
      + tile * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS;
    const low = words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskLow]!;
    const high = words[at + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE.laneMaskHigh]!;
    // A one-cell-deep extrusion can publish only the first four lanes of each
    // y row; every z>0 lane and therefore the complete high word stays clear.
    assert.equal(high, 0);
    assert.equal(low & 0xffff_0000, 0);
  }
});

test("production physical leaf gaps survive projected WDR, TEI and HTP banks", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const accepted = slice.topology.accepted;
  const shifted = compileSliceTopology(accepted.bricks.map(brick => ({ ...brick,
    id: brick.id + 6, key: brick.key + 6 })), accepted.dimensions,
  accepted.generation, 0.5, accepted.boundaryModes);
  const authority = createSliceRuntimeAuthority(shifted);
  assert.equal(authority.leafCapacity, 22);
  assert.deepEqual(Array.from(authority.accepted.compactLeafToStable),
    shifted.bricks.map(brick => brick.id));
  assert.ok(Array.from(authority.accepted.transportPacketSchedule)
    .every(packet => Math.floor(packet / 64) >= 6));
  assert.equal(authority.accepted.stableCellOrder[0], 6 * 64);
  const htp = authority.accepted.hotTopology;
  const packed = htp.words[htp.layout.cellBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution]!;
  assert.equal(packed >>> 5, 6);
  assert.equal(sliceRuntimeNextLeaf(authority.accepted), 22);
});

test("candidate banks commit atomically and cancellation preserves the accepted slot", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const accepted = slice.topology.accepted;
  const authority = createSliceRuntimeAuthority(accepted,
    { leafCapacity: accepted.bricks.length + 2 });
  const candidateBricks = accepted.bricks.map((brick, index) => index === 0
    ? { id: brick.id, key: brick.key, coordinate: brick.coordinate,
      spanBricks: brick.spanBricks, resolution: 2 as const, active: brick.active }
    : brick);
  const candidate = compileSliceTopology(candidateBricks, accepted.dimensions,
    accepted.generation + 1, 0.5, accepted.boundaryModes);
  const staged = stageSliceRuntimeAuthority(authority, candidate);
  assert.equal(staged.receipt.phase, "candidate-ready");
  assert.equal(staged.receipt.acceptedGeneration, accepted.generation);
  assert.equal(staged.receipt.candidateGeneration, candidate.generation);
  assert.equal(staged.acceptedSlot, 0);
  assert.equal(staged.receipt.candidateSlot, 1);
  assert.equal(staged.accepted.topology, accepted);
  assert.equal(staged.candidate?.topology, candidate);

  const cancelled = cancelSliceRuntimeAuthority(staged);
  assert.equal(cancelled.accepted.topology, accepted);
  assert.equal(cancelled.acceptedSlot, 0);
  assert.equal(cancelled.candidate, undefined);

  const committed = commitSliceRuntimeAuthority(staged);
  assert.equal(committed.accepted.topology, candidate);
  assert.equal(committed.acceptedSlot, 1);
  assert.equal(committed.candidate, undefined);
  assert.equal(committed.receipt.acceptedGeneration, candidate.generation);
});

test("a candidate beyond fixed WDR capacity fail-closes without replacing accepted banks", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const accepted = slice.topology.accepted;
  const authority = createSliceRuntimeAuthority(accepted);
  const extra = { id: accepted.bricks.length, key: accepted.bricks.length,
    coordinate: [4, 0] as const, resolution: 1 as const, active: false };
  const candidate = compileSliceTopology([...accepted.bricks, extra],
    accepted.dimensions, accepted.generation + 1, 0.5,
    accepted.boundaryModes);
  const failed = stageSliceRuntimeAuthority(authority, candidate);
  assert.equal(failed.receipt.fault, 1);
  assert.equal(failed.accepted.topology, accepted);
  assert.equal(failed.candidate, undefined);
  assert.equal(failed.acceptedSlot, 0);
});

test("a dynamic WDR leaf retires after publication and reuses its stable slot", () => {
  const authored = compileSliceTopology([{ id: 0, key: 0, coordinate: [0, 0],
    resolution: 1, active: true }], [24, 8], 1, 0.5,
  { negativeX: "closed", positiveX: "closed", negativeY: "closed", positiveY: "closed" });
  const initial = createSliceRuntimeAuthority(authored, { leafCapacity: 2 });
  const allocatedTopology = compileSliceTopology([...authored.bricks,
    { id: 1, key: 1, coordinate: [1, 0] as const, resolution: 1 as const,
      active: false }], authored.dimensions, 2, 0.5, authored.boundaryModes);
  const allocated = commitSliceRuntimeAuthority(
    stageSliceRuntimeAuthority(initial, allocatedTopology));
  const retired = releaseSliceRuntimeLeaves(allocated, [1]);
  const image = retired.authority.accepted;
  const wdr = image.worldDirectoryWords, wl = image.worldDirectoryLayout;
  const leafAt = wl.leafBaseWords + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
  assert.deepEqual(Array.from(retired.receipt.releasedLeafIds), [1]);
  assert.deepEqual(Array.from(retired.receipt.rejectedLeafIds), []);
  assert.equal(retired.receipt.fault, 0);
  assert.equal(retired.receipt.liveLeafCount, 1);
  assert.equal(retired.receipt.freeLeafCount, 1);
  assert.equal(wdr[leafAt + SPARSE_CM12_WORLD_DIRECTORY_LEAF.generation], 0xffff_ffff);
  assert.equal(wdr[wl.freeListBaseWords], 1);
  const tombstones = Array.from({ length: wl.capacity }, (_, slot) =>
    wl.entryBaseWords + slot * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS)
    .filter(at => wdr[at + SPARSE_CM12_WORLD_DIRECTORY_ENTRY.state] === 3
      && wdr[at + SPARSE_CM12_WORLD_DIRECTORY_ENTRY.leaf] === 1);
  assert.equal(tombstones.length, 1);
  const tei = image.transportExecutionImage.layout;
  for (const bank of [0, 1] as const) {
    const teiLeaf = tei.slotLeafBaseOffsets[bank]
      + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS;
    assert.equal(image.transportExecutionImage.words[teiLeaf
      + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF.flags] >>> 31, 0);
    for (let local = 0; local < tei.packetsPerLeaf; local += 1) {
      const at = tei.slotPacketBaseOffsets[bank]
        + (tei.packetsPerLeaf + local) * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
      assert.equal(image.transportExecutionImage.words[at
        + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET.first], 0xffff_ffff);
    }
  }

  const reusedTopology = compileSliceTopology([...authored.bricks,
    { id: 1, key: 2, coordinate: [2, 0] as const, resolution: 1 as const,
      active: true }], authored.dimensions, 3, 0.5, authored.boundaryModes);
  const reused = commitSliceRuntimeAuthority(
    stageSliceRuntimeAuthority(retired.authority, reusedTopology));
  assert.equal(reused.accepted.compactLeafToStable[1], 1);
  const reusedLeafAt = reused.accepted.worldDirectoryLayout.leafBaseWords
    + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
  assert.equal(reused.accepted.worldDirectoryWords[reusedLeafAt
    + SPARSE_CM12_WORLD_DIRECTORY_LEAF.x], 2);
  assert.equal(reused.accepted.worldDirectoryWords[reusedLeafAt
    + SPARSE_CM12_WORLD_DIRECTORY_LEAF.generation], 3);
});
