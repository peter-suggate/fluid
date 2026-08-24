import assert from "node:assert/strict";
import test from "node:test";

import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  cloneSparseCM12TilePoolReceiver,
  createSparseCM12TileClonePool,
  sparseCM12TileClonePoolLookup,
  sparseCM12TileCloneSeedsFromBricks,
  SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY,
  SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY,
  SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD,
  SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES,
  SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD,
  SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES,
  SPARSE_CM12_TILE_CLONE_POOL_HEADER,
  SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS,
  SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT,
  SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS,
} from "../lib/methods/adaptive-mass/sparse-cm12-tile-clone-pool";
import { adaptiveMassPresentationDimensionsForScene } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

test("tile residency is invariant to vast empty-world extent", () => {
  const scene = createSparseCM12LongDamBreakScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const localSeeds = sparseCM12TileCloneSeedsFromBricks(atlas.bricks);
  assert.equal(localSeeds.length, 80);

  // Put the same physical dam near the far corner of a hypothetical world
  // containing 8e27 addressable brick coordinates. Extent is deliberately not
  // an input to the pool: empty coordinate space has no allocation footprint.
  const offset = [1_000_000_000, -1_000_000_000, 900_000_000] as const;
  const remoteSeeds = localSeeds.map((seed) => ({
    ...seed,
    coordinate: [
      seed.coordinate[0] + offset[0],
      seed.coordinate[1] + offset[1],
      seed.coordinate[2] + offset[2],
    ] as [number, number, number],
  }));
  const local = createSparseCM12TileClonePool(localSeeds);
  const remote = createSparseCM12TileClonePool(remoteSeeds);
  const header = SPARSE_CM12_TILE_CLONE_POOL_HEADER;
  const hypotheticalWorldTiles = 2_000_000_000n ** 3n;

  assert.deepEqual(remote.layout, local.layout,
    "moving into a vastly larger coordinate space must not alter allocation");
  assert.equal(remote.words[header.residentCount], 80);
  assert.equal(remote.words[header.highWaterMark], 80);
  const receipt = SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT;
  const completePageCount = (pool: typeof remote) => Array.from(
    { length: pool.layout.capacity }, (_, slot) => pool.pageReceipts[
      slot * SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS + receipt.flags
    ]!,
  ).filter((flags) => (flags & SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS)
    === SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS).length;
  assert.equal(completePageCount(remote), 80,
    "only resident coordinates may own initialized execution pages");
  assert.ok(BigInt(remote.layout.capacity) * 1_000_000_000_000n
    < hypotheticalWorldTiles,
  "physical capacity must be unrelated to logical-world volume");

  const source = remoteSeeds.reduce((rightmost, seed) =>
    seed.coordinate[0] > rightmost.coordinate[0] ? seed : rightmost);
  const receiver = [source.coordinate[0] + 1, source.coordinate[1],
    source.coordinate[2]] as [number, number, number];
  assert.equal(sparseCM12TileClonePoolLookup(remote, receiver), undefined,
    "an empty coordinate must have no resident page");

  const advanced = cloneSparseCM12TilePoolReceiver(remote, receiver, 8);
  assert.equal(advanced.layout.totalBytes, remote.layout.totalBytes,
    "a clone consumes one reserved physical page, not world-sized storage");
  assert.equal(advanced.words[header.residentCount], 81);
  assert.equal(advanced.words[header.highWaterMark], 81);
  assert.equal(advanced.words[header.cloneCount], 1);
  const receiverSlot = sparseCM12TileClonePoolLookup(advanced, receiver);
  assert.notEqual(receiverSlot, undefined);
  assert.equal(completePageCount(advanced), 81);
  const page = receiverSlot! * SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS;
  assert.equal(advanced.pageReceipts[page + receipt.resolution], 8);
  assert.equal(advanced.pageReceipts[page + receipt.cellCount], 512);
  assert.equal(advanced.pageReceipts[page + receipt.faceCount], 1_728);
  assert.equal(advanced.pageReceipts[page + receipt.incidenceCount], 3_072);
  assert.equal(advanced.pageReceipts[page + receipt.internalPressureEdgeCount], 1_344);
  const gammaBase = receiverSlot! * SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES
    * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY
    + SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD.gammaA
    * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY;
  const openFaceBase = receiverSlot! * SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES
    * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY
    + SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD.openFraction
    * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY;
  assert.equal(advanced.cellFields[gammaBase], 1,
    "a cloned empty receiver must start with the gamma identity");
  assert.equal(advanced.faceFields[openFaceBase], 1,
    "a cloned empty receiver must start with open sparse-air faces");
});
