import assert from "node:assert/strict";
import test from "node:test";
import { uniformMixedRemapCapacity, UNIFORM_REMAP_PLANE_WORDS } from "../lib/methods/uniform/uniform-mixed-remap-capacity";

test("large terrain remaps hold every unique 4h corner, beyond the former 1,024 limit", () => {
  const dimensions = [128, 128, 128] as const;
  const tiles = 32 ** 3;
  const capacity = uniformMixedRemapCapacity(dimensions, tiles, 128 * 1024 ** 2, true);
  const corners = new Set<string>();
  for (let z = 0; z < 32; z++) for (let x = 0; x < 32; x++) {
    // A broad, one-tile-deep terrain layer alone has 2,178 corners.
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++)
      corners.add(`${x + dx},${dy},${z + dz}`);
  }
  assert.ok(corners.size > 1024);
  assert.ok(capacity.corners >= corners.size);
  assert.equal(capacity.corners, 33 ** 3);
  assert.equal(capacity.bytes, 4 * (4 + tiles + 2 * 33 ** 3),
    "plane-fit partials stay in workgroup memory, not a per-corner global allocation");
  assert.ok(capacity.bytes <= 128 * 1024 ** 2);
});

test("sparse remap capacity follows stored tiles and respects the binding limit", () => {
  const dimensions = [4096, 4096, 4096] as const;
  const sparse = uniformMixedRemapCapacity(dimensions, 8, 128 * 1024 ** 2, true);
  assert.equal(sparse.corners, 8 * 8);
  const maxBytes = 65536;
  const limited = uniformMixedRemapCapacity(dimensions, 8192, maxBytes, true);
  assert.ok(limited.bytes <= maxBytes);
  assert.ok(limited.bytes + 4 * UNIFORM_REMAP_PLANE_WORDS > maxBytes);
  assert.equal(uniformMixedRemapCapacity(dimensions, 512, maxBytes, false).corners, 1);
  assert.throws(() => uniformMixedRemapCapacity(dimensions, 512, 16, true), /Device cannot hold/);
});
