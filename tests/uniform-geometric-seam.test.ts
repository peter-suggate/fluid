import assert from "node:assert/strict";
import test from "node:test";
import { geometricSeamRows, transportSeamReference, seamLayout, type Triple } from "./helpers/uniform-geometric-seam";

const sum = (a: ArrayLike<number>) => Array.from(a).reduce((s, x) => s + x, 0);
for (const mode of ["mixed", "fine", "coarse"] as const) for (let axis = 0; axis < 3; axis++) {
  test(`${mode}, axis ${axis}: geometric transport identity and large-step donor conservation`, () => {
    const layout = seamLayout(axis, mode);
    const still = geometricSeamRows(layout, () => [0, 0, 0]);
    const volume = Float64Array.from(still.cells, (c, i) => c.capacity * (i % 7) / 3);
    assert.deepEqual(transportSeamReference(still, volume), volume, "includes retained excess above capacity");
    for (const sign of [-1, 1]) for (const distance of [.25, 2.5, 12]) {
      const delta = [0, 0, 0]; delta[axis] = sign * distance;
      const rows = geometricSeamRows(layout, () => delta as unknown as Triple);
      const result = transportSeamReference(rows, volume);
      assert.ok(Math.abs(sum(result) - sum(volume)) < 1e-10);
      assert.ok(result.every(v => Number.isFinite(v) && v >= 0));
      if (distance === 12) assert.deepEqual(result, volume, "unsampled donors fall back to themselves");
    }
  });
}
for (let axis = 0; axis < 3; axis++) test(`axis ${axis}: liquid crosses the seam in both directions`, () => {
  for (const sign of [-1, 1]) {
    const delta = [0, 0, 0]; delta[axis] = sign * .25;
    const rows = geometricSeamRows(seamLayout(axis), () => delta as unknown as Triple);
    const volumes = rows.cells.map(c => (sign > 0 ? c.width === 1 : c.width === 4) ? c.capacity : 0);
    const result = transportSeamReference(rows, volumes);
    assert.ok(rows.cells.some((c, i) => (sign > 0 ? c.width === 4 : c.width === 1) && result[i]! > 0));
    assert.ok(Math.abs(sum(result) - sum(volumes)) < 1e-10);
  }
});
test("coarse departure footprint enumerates more than eight fine donors", () => {
  const rows = geometricSeamRows(seamLayout(0), () => [2.5, .25, .25]);
  assert.ok(rows.cells.some((c, i) => c.width === 4 && rows.offsets[i + 1]! - rows.offsets[i]! > 9));
});

test("fixed 8/27/125 overlap patterns match variable-row geometry, including long traces", async () => {
  const { fixedGeometricSeamRows } = await import('./helpers/uniform-geometric-seam');
  for (const mode of ['fine', 'coarse', 'mixed'] as const) for (let axis = 0; axis < 3; axis++) for (const distance of [-5.25, -.25, 0, .25, 5.25]) {
    const delta = [.25, -.5, .75]; delta[axis] = distance;
    const layout = seamLayout(axis, mode), displacement = () => delta as unknown as Triple;
    const expected = geometricSeamRows(layout, displacement), actual = fixedGeometricSeamRows(layout, displacement);
    for (let i = 0; i < actual.cells.length; i++) {
      const aggregate = (rows: typeof actual) => {
        const result = new Float64Array(rows.cells.length);
        for (let e = rows.offsets[i]!; e < rows.offsets[i + 1]!; e++) result[rows.donors[e]!] += rows.weights[e]!;
        return result;
      };
      assert.deepEqual(aggregate(actual), aggregate(expected), `${mode} axis ${axis} distance ${distance} row ${i}`);
    }
    const volume = actual.cells.map((c, i) => c.capacity * (i % 7) / 3);
    const a = transportSeamReference(actual, volume), b = transportSeamReference(expected, volume);
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]!) < 1e-10));
  }
});
test("fixed-pattern edge capacity never exceeds the existing finest edge arena", async () => {
  const { uniformVolumeStencilBytes } = await import('../lib/methods/uniform/uniform-volume-stencil');
  assert.equal(uniformVolumeStencilBytes(1), 40);
  assert.equal(uniformVolumeStencilBytes(2), 116);
  assert.equal(uniformVolumeStencilBytes(4), 508);
  for (const width of [1, 2, 4] as const) assert.ok((4 / width) ** 3 * uniformVolumeStencilBytes(width) <= 64 * 40);
});

test("graded 1h/2h/4h ownership preserves the fixed-stencil transport oracle", async () => {
  const { createUniformMixedLayout } = await import('../lib/methods/uniform/uniform-mixed-layout');
  const { fixedGeometricSeamRows } = await import('./helpers/uniform-geometric-seam');
  const layout = createUniformMixedLayout({ dimensions: [16, 8, 8], cellSize_m: [1, 2, .5], origin_m: { x: 0, y: 0, z: 0 } }, [{
    id: 'fine', rule: 'minimum-cell-size', minimumCellSize_cells: 1, maximumCellSize_cells: 1,
    min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 4, y: 16, z: 4 },
  }], true, 4);
  for (const distance of [-6.25, -.25, 0, .25, 6.25]) {
    const displacement = () => [distance, .25, -.5] as const;
    const expected = geometricSeamRows(layout, displacement), fixed = fixedGeometricSeamRows(layout, displacement);
    assert.deepEqual(new Set(fixed.cells.map(c => c.width)), new Set([1, 2, 4]));
    const volume = fixed.cells.map((c, i) => c.capacity * ((i * 13) % 31) / 31);
    const a = transportSeamReference(fixed, volume), b = transportSeamReference(expected, volume);
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]!) < 1e-10));
    assert.ok(Math.abs(sum(a) - sum(volume)) < 1e-10);
  }
});
