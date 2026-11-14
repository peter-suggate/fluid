import assert from "node:assert/strict";
import test from "node:test";
import {
  BRICKS, BULK_LISTS, BX, BY, BZ, CLASS_BULK, CLASS_DEEP, DEEP_ROW_LIMIT, NBULK, THROAT_Y0, THROAT_Z0, WALL_X0,
  bakeSeep, brickId, bulkBrick, throatBlockBrick,
} from "../lib/methods/seep/seep-layout";

test("seep BULK (4h composite) bricks are deep reservoir bricks clear of the throat approach", () => {
  assert.equal(NBULK, 30 * 23 * 8 - 8);
  assert.equal(new Set(BULK_LISTS.bulk).size, NBULK);
  for (const b of BULK_LISTS.bulk) {
    const bx = b % BX, by = Math.floor(b / BX) % BY, bz = Math.floor(b / (BX * BY));
    assert.ok(bulkBrick(bx, by, bz));
    assert.ok(bx * 8 + 8 <= WALL_X0 && by * 8 + 8 <= DEEP_ROW_LIMIT, `brick ${b} outside the deep reservoir`);
    assert.ok(!throatBlockBrick(bx, by, bz));
  }
  // The throat's approach stays fine (h), as the reference pins the throat (+-4 cells) at h: no cell
  // within 6 cells of the 2x2 mouth (x 240, y 64..65, z 32..33), nor 16 cells upstream of it, is BULK.
  const tby = THROAT_Y0 >> 3, tbz = THROAT_Z0 >> 3;
  for (let x = WALL_X0 - 16; x < WALL_X0; x++) for (let y = THROAT_Y0 - 6; y < THROAT_Y0 + 8; y++)
    for (let z = THROAT_Z0 - 6; z < THROAT_Z0 + 8; z++)
      assert.ok(!bulkBrick(x >> 3, y >> 3, z >> 3), `throat approach cell ${x},${y},${z} is BULK`);
  // Shadow bricks (fine velocity copies next to fine cells) are a subset of BULK.
  const bulk = new Set(BULK_LISTS.bulk);
  assert.ok(BULK_LISTS.shadow.length > 0 && BULK_LISTS.shadow.every(b => bulk.has(b)));
  assert.ok(BULK_LISTS.shadow.includes(brickId(27, tby, tbz)), "the brick facing the throat block keeps a shadow");
  assert.ok(!BULK_LISTS.shadow.includes(brickId(0, 0, 0)), "an interior corner brick needs no shadow");
});

test("the seep bake marks exactly the BULK bricks CLASS_BULK, all of them DEEP", () => {
  const { brickClass } = bakeSeep();
  let count = 0;
  for (let b = 0; b < BRICKS; b++) {
    const bx = b % BX, by = Math.floor(b / BX) % BY, bz = Math.floor(b / (BX * BY));
    const isBulk = (brickClass[b] & CLASS_BULK) !== 0;
    assert.equal(isBulk, bulkBrick(bx, by, bz) && bz < BZ, `brick ${b}`);
    if (isBulk) { count++; assert.ok(brickClass[b] & CLASS_DEEP); }
  }
  assert.equal(count, NBULK);
});
