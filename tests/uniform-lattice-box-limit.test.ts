import assert from "node:assert/strict";
import test from "node:test";
import { uniformLatticeAddress, uniformLatticeBox } from "../lib/methods/uniform/uniform-lattice-address";

test("a box has no page limit of its own: the page table's window is the address's", () => {
  const box = uniformLatticeBox([4096, 256, 4096]);
  assert.deepEqual(box.pages, [128, 8, 128]); assert.equal(box.boxTiles, 1024 * 64 * 1024);
  assert.throws(() => uniformLatticeAddress([4096, 256, 4096], true, { blocks: 64 }), /exceeds 62 pages/);
  assert.equal(uniformLatticeAddress([1984, 64, 64], false).blocks, 62 * 2 * 2 + 2);
});
