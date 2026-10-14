import assert from "node:assert/strict";
import { test } from "node:test";
import { particleResidencyTravel, residencyDirectionRadii, residencyOccupancyMeets } from "../lib/methods/uniform/uniform-local-residency";

test("particle allocation follows accepted motion, with local allowance for ballistic contact turns", () => {
  // A large FLIP residual is momentum, not the velocity used to move this coupled particle.
  assert.deepEqual(particleResidencyTravel([1000, -2000, 3000], [1, -2, 0], false, [1, 1, 1]), [0, 2, 0, 1, 0, 0]);
  // A 3-4-5 ballistic velocity can turn along a slope; no axis may outrun its speed.
  assert.deepEqual(particleResidencyTravel([3, -4, 0], [1000, 0, 0], true, [1, 1, 2]), [5, 5, 10, 5, 5, 10]);
});

test("signed allocation travel rounds outward and retains the full stencil margin", () => {
  assert.deepEqual(residencyDirectionRadii([-8.01, 0, -1], [0, 12, 1], [1, 1, 2]), [8, 5, 6, 5, 8, 6]);
  assert.deepEqual(residencyDirectionRadii([0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 40.01, 8, 0, 0]), [5, 5, 16, 7, 5, 5]);
  assert.deepEqual(residencyDirectionRadii([0, 0, 0], [0, 0, 0], [1, 1, 1]), [5, 5, 5, 5, 5, 5]);
  // Overflow stays above the pool's cap; it must not wrap into a small reach.
  assert.equal(residencyDirectionRadii([-1e30, 0, 0], [0, 0, 0], [1, 1, 1])[0], 255);
});

test("disconnected seeds do not fill their bounding box", () => {
  const mask = [1, 0x80000000]; // opposite corners
  assert.equal(residencyOccupancyMeets(mask, [0, 0, 0, 0, 0, 0], [1, 1, 1], [2, 2, 2]), false);
  assert.equal(residencyOccupancyMeets(mask, [0, 0, 0, 0, 0, 0], [3, 3, 3], [3, 3, 3]), true);
  // A +x sweep reaches +x only. The inverse test must swap the radii.
  assert.equal(residencyOccupancyMeets([1, 0], [0, 0, 0, 8, 0, 0], [8, 0, 0], [15, 7, 7]), true);
  assert.equal(residencyOccupancyMeets([1, 0], [0, 0, 0, 8, 0, 0], [-8, 0, 0], [-1, 7, 7]), false);
});

test("occupancy queries equal an independent enumeration of every swept seed", () => {
  let random = 1234567;
  const next = () => (random = (Math.imul(random, 1664525) + 1013904223) >>> 0);
  for (let trial = 0; trial < 20000; trial++) {
    const mask = [next() & next(), next() & next()];
    const radii = Array.from({ length: 6 }, () => next() % 33);
    const low = Array.from({ length: 3 }, () => (next() % 21 - 10) * 4);
    const edge = (next() & 1) ? 4 : 8, high = low.map(v => v + edge - 1);
    let expected = false;
    for (let lane = 0; lane < 64; lane++) {
      if (((mask[lane >>> 5]! >>> (lane & 31)) & 1) === 0) continue;
      const point = [lane & 3, (lane >>> 2) & 3, lane >>> 4];
      if (point.every((v, k) => v - radii[k]! <= high[k]! && v + radii[k + 3]! >= low[k]!)) expected = true;
    }
    assert.equal(residencyOccupancyMeets(mask, radii, low, high), expected, `trial ${trial}`);
  }
});
