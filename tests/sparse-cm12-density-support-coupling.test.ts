import assert from "node:assert/strict";
import test from "node:test";
import { compileDensitySupportCoupling, applyDensitySupportCoupling, densitySupportGeometryKey, assertDensitySupportCouplingSupport, type DensityBernsteinSupportBox } from "../lib/methods/adaptive-mass/sparse-cm12-density-support-coupling";
import { compileDensityNativeGeometry, type DensityNativeVec3 } from "../lib/methods/adaptive-mass/sparse-cm12-density-native-geometry";
import type { SparseAtlasCompositeCell } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
const stamp = { topologyGeneration: 3, boundaryGeneration: 7, supportGeneration: 11 };
function geometry(boxes: readonly DensityBernsteinSupportBox[]) {
  return compileDensityNativeGeometry({ ...stamp, fineCellWidth: 1, rows: [], cells: boxes.map(box => ({
    id: box.id, stableLeafId: box.id, minimumFine: box.lower, maximumFine: box.upper,
  } as SparseAtlasCompositeCell)) });
}
const vec = (v: number[]) => v as unknown as DensityNativeVec3;
const box = (id: number, lower: number[], upper: number[]): DensityBernsteinSupportBox => ({ id, lower: vec(lower), upper: vec(upper) });
function close(a: number, b: number, tolerance = 2e-12) { assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)), `${a} != ${b}`); }
function evaluate(controls: ArrayLike<number>, p: DensityNativeVec3, support: DensityBernsteinSupportBox) {
  const basis = p.map((v, axis) => { const t = (v - support.lower[axis]!) / (support.upper[axis]! - support.lower[axis]!); return [(1 - t) ** 2, 2 * t * (1 - t), t * t]; });
  let sum = 0;
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) sum += controls[x + 3 * y + 9 * z]! * basis[0]![x]! * basis[1]![y]! * basis[2]![z]!;
  return sum;
}
function gaussMean(domain: DensityBernsteinSupportBox, support: DensityBernsteinSupportBox, controls: ArrayLike<number>) {
  let total = 0;
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    const signs = [x, y, z];
    const point = vec(domain.lower.map((lo, axis) => (lo + domain.upper[axis]!) / 2 + signs[axis]! * (domain.upper[axis]! - lo) / (2 * Math.sqrt(3))));
    total += evaluate(controls, point, support) / 8;
  }
  return total;
}
test("all 27 Bernstein modes integrate clipped anisotropic boxes against independent Gauss quadrature", () => {
  const support = box(91, [-3, 2, -7], [5, 3, 9]);
  const native = box(72, [-2.7, 2.02, -3.3], [4.8, 2.73, 8.999]);
  const coupling = compileDensitySupportCoupling({ ...stamp, geometry: geometry([native]), supports: [support] });
  assert.equal(coupling.axisMoments.length, 9);
  assert.deepEqual([...coupling.supportIndices], [0]);
  assert.equal(coupling.receipt.incompleteCells, 0);
  for (let mode = 0; mode < 27; mode++) {
    const controls = Array.from({ length: 27 }, (_, k) => Number(k === mode));
    const means = applyDensitySupportCoupling(coupling, { ...stamp, supportGeometryKey: coupling.supportGeometryKey, controls: () => controls });
    close(means[0]!, gaussMean(native, support, controls));
  }
});
test("physics repartition integrates unchanged retained controls and conserves local amount", () => {
  const supports = [box(44, [0, 0, 0], [1, 1, 1]), box(77, [1, 0, 0], [2, 1, 1])];
  const controls = new Map(supports.map(s => [s.id, Float64Array.from({ length: 27 }, (_, i) => 0.1 + ((i * 7 + s.id) % 29) / 10)]));
  const before = JSON.stringify([...controls].map(([id, c]) => [id, [...c]]));
  const whole = [box(900, [0, 0, 0], [2, 1, 1])];
  const split = [box(901, [0, 0, 0], [0.7, 1, 1]), box(902, [0.7, 0, 0], [1.3, 1, 1]), box(903, [1.3, 0, 0], [2, 1, 1])];
  const integrate = (cells: DensityBernsteinSupportBox[]) => {
    const coupling = compileDensitySupportCoupling({ ...stamp, geometry: geometry(cells), supports });
    const means = applyDensitySupportCoupling(coupling, { ...stamp, supportGeometryKey: coupling.supportGeometryKey, controls: id => controls.get(id)! });
    return means.reduce((mass, mean, i) => mass + mean * coupling.cellVolumes[i]!, 0);
  };
  close(integrate(whole), integrate(split));
  close(integrate(whole), [...controls.values()].reduce((sum, c) => sum + c.reduce((a, b) => a + b, 0) / 27, 0));
  assert.equal(JSON.stringify([...controls].map(([id, c]) => [id, [...c]])), before);
});
test("missing support, overlapping ownership, unknown clipping and stale epochs fail explicitly", () => {
  const native = box(5, [0, 0, 0], [2, 1, 1]), supports = [box(9, [0, 0, 0], [1, 1, 1])];
  const args = { ...stamp, geometry: geometry([native]), supports };
  assert.throws(() => compileDensitySupportCoupling(args), /does not cover/);
  const partial = compileDensitySupportCoupling({ ...args, requireCoverage: false });
  assert.equal(partial.coveredVolumes[0], 1);
  assert.equal(partial.receipt.maximumUncoveredFraction, 0.5);
  assert.throws(() => compileDensitySupportCoupling({ ...args, supports: [native, ...supports] }), /overlapping/);
  assert.throws(() => compileDensitySupportCoupling({ ...args, supports: [native], requireFullyOpen: true }), /fully-open/);
  assert.throws(() => compileDensitySupportCoupling({ ...args, boundaryGeneration: 8 }), /stale/);
  assert.throws(() => applyDensitySupportCoupling(partial, { ...stamp, supportGeometryKey: partial.supportGeometryKey, supportGeneration: 12, controls: () => [] }), /stale/);
});
test("spatial compilation stays sparse with increasing empty extent and macro width", () => {
  const supports = Array.from({ length: 1024 }, (_, i) => box(i, [i * 1048576, 0, 0], [i * 1048576 + 1048576, 1048576, 1048576]));
  const coupling = compileDensitySupportCoupling({ ...stamp, geometry: geometry(supports), supports });
  assert.equal(coupling.receipt.intersections, 1024);
  assert.equal(coupling.receipt.bvhNodes, 2047);
  assert.ok(coupling.receipt.boxTests < 1024 * 60);
  assert.equal(coupling.axisMoments.length, 9 * 1024);
  const means = applyDensitySupportCoupling(coupling, { ...stamp, supportGeometryKey: coupling.supportGeometryKey, controls: () => new Float64Array(27).fill(2.5) });
  for (const mean of means) close(mean, 2.5);
});

test("matching epoch numbers cannot authorize reordered or changed retained support", () => {
  const supports = [box(9, [0, 0, 0], [1, 1, 1]), box(17, [1, 0, 0], [2, 1, 1])];
  const coupling = compileDensitySupportCoupling({ ...stamp, geometry: geometry(supports), supports });
  assert.equal(coupling.supportGeometryKey, densitySupportGeometryKey(supports));
  assert.doesNotThrow(() => assertDensitySupportCouplingSupport(coupling, supports.map(s => ({ ...s, lower: [...s.lower], upper: [...s.upper] })), stamp.supportGeneration));
  assert.throws(() => assertDensitySupportCouplingSupport(coupling, [...supports].reverse(), stamp.supportGeneration), /identity mismatch/);
  assert.throws(() => assertDensitySupportCouplingSupport(coupling, [supports[0]!, { ...supports[1]!, upper: [2.0000000000000004, 1, 1] }], stamp.supportGeneration), /identity mismatch/);
  assert.throws(() => assertDensitySupportCouplingSupport(coupling, [{ ...supports[0]!, id: 10 }, supports[1]!], stamp.supportGeneration), /identity mismatch/);
  assert.throws(() => applyDensitySupportCoupling(coupling, { ...stamp, supportGeometryKey: densitySupportGeometryKey([...supports].reverse()), controls: () => new Float64Array(27) }), /identity mismatch/);
});
