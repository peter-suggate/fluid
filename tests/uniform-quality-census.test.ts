import assert from "node:assert/strict";
import test from "node:test";
import {uniformQualityCensus} from "../tools/uniform-quality-census";

test("quality metrics describe the same physical slab across h/4h ownership", () => {
  const d = [8, 4, 4] as const, volume = new Float32Array(128), phi = new Float32Array(9 * 5 * 5);
  for (let z = 0; z <= 4; z++) for (let y = 0; y <= 4; y++) for (let x = 0; x <= 8; x++) phi[x + 9 * (y + 5 * z)] = x - 4;
  for (let i = 0; i < volume.length; i++) volume[i] = i % 8 < 4 ? 1 : 0;
  const fine = uniformQualityCensus(d, new Uint32Array([0x80000000, 0x80000000]), volume, phi);
  const coarse = uniformQualityCensus(d, new Uint32Array(2), volume, phi);
  for (const c of [fine, coarse]) {
    assert.equal(c.mass, 64); assert.equal(c.excess, 0); assert.equal(c.negative, 0);
    assert.deepEqual(c.centroid_cells, [2, 2, 2]); assert.equal(c.massFront_cells.p99, 3.96);
  }
  assert.deepEqual(coarse.projection, fine.projection); assert.deepEqual(coarse.phiSlice, fine.phiSlice);
});

test("quality census reports empty, invalid and overfilled states without hiding them", () => {
  const d = [4, 4, 4] as const, v = new Float32Array(64), phi = new Float32Array(125).fill(1);
  const tiles = new Uint32Array(1);
  const empty = uniformQualityCensus(d, tiles, v, phi);
  assert.equal(empty.centroid_cells, null); assert.equal(empty.massFront_cells.p99, null);
  v[0] = 1.25;
  const over = uniformQualityCensus(d, tiles, v, phi);
  assert.equal(over.mass, 80); assert.equal(over.excess, 16); assert.equal(over.massInPhiAir, 80);
  v[0] = -.25; assert.equal(uniformQualityCensus(d, tiles, v, phi).negative, 16);
  v[0] = NaN; assert.equal(uniformQualityCensus(d, tiles, v, phi).nonfinite, 1);
});
