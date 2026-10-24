import assert from "node:assert/strict";
import test from "node:test";
import { UNIFORM_BLOCK_GRID_BLOCKS, UNIFORM_OPEN_WORLD_MEMORY_MiB, uniformLatticeAddress, uniformLatticeBox, uniformPagePoolBlocksForMemory } from "../lib/methods/uniform/uniform-lattice-address";

test("a box has no page limit of its own: the page table's window is the address's", () => {
  const box = uniformLatticeBox([4096, 256, 4096]);
  assert.deepEqual(box.pages, [128, 8, 128]); assert.equal(box.boxTiles, 1024 * 64 * 1024);
  assert.throws(() => uniformLatticeAddress([4096, 256, 4096], true, { blocks: 64 }), /exceeds 62 pages/);
  assert.equal(uniformLatticeAddress([1984, 64, 64], false).blocks, 62 * 2 * 2 + 2);
});

test("a box with more pages than the block grid holds pools its resident pages, not its own", () => {
  // 48x8x48 pages: the box's default (18434 blocks) is past the grid; a corner of 4x2x4 pages is resident.
  const [px, py, pz] = [48, 8, 48], air: [number, number, number][] = [];
  for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = 0; x < px; x++) if (x >= 4 || y >= 2 || z >= 4) air.push([x, y, z]);
  assert.ok(px * py * pz + 2 > UNIFORM_BLOCK_GRID_BLOCKS);
  const address = uniformLatticeAddress([32 * px, 32 * py, 32 * pz], false, { air });
  assert.equal(address.blocks, uniformPagePoolBlocksForMemory(UNIFORM_OPEN_WORLD_MEMORY_MiB));
  assert.equal(address.liveTiles, 512 * 4 * 2 * 4);
  // A box the grid holds keeps its own default.
  assert.equal(uniformLatticeAddress([1024, 256, 256], false).blocks, 32 * 8 * 8 + 2);
  assert.throws(() => uniformLatticeAddress([32 * px, 32 * py, 32 * pz], false), /pool/);
});
