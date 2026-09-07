import assert from "node:assert/strict";
import test from "node:test";
import { applyStencil, compileStencil, packetCost, type Donor } from "../tools/implicit-density/compiled-stencil";
import { type Box, type Frame, type Vec3 } from "../tools/implicit-density/field";

const frame: Frame = { origin: [11, -7, 3], scale: [2, 0.3, 5] };
const vec = (xs: number[]) => xs as unknown as Vec3;
const physical = (p: Vec3) => vec(p.map((x, a) => frame.origin[a] + x * frame.scale[a]));

// Test-owned tensor Gauss rule: exact for degree <= 3 in each coordinate.
// Neither donor data nor verification use meanBasis/meanPolynomial.
function boxMean(box: Box, f: (p: Vec3) => number): number {
  let total = 0;
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    const signs = [x, y, z];
    total += f(vec(box.lower.map((lo, a) => (lo + box.upper[a]) / 2
      + signs[a] * (box.upper[a] - lo) / (2 * Math.sqrt(3))))) / 8;
  }
  return total;
}
function value(coefficients: readonly number[], p: Vec3): number {
  const [x, y, z] = p.map((v, a) => (v - frame.origin[a]) / frame.scale[a]);
  return coefficients[0] + coefficients[1] * x + coefficients[2] * y
    + coefficients[3] * z + coefficients[4] * x * x + coefficients[5] * y * y
    + coefficients[6] * z * z + coefficients[7] * x * y
    + coefficients[8] * x * z + coefficients[9] * y * z;
}
function close(actual: number, expected: number, tolerance = 3e-11): void {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}; error=${Math.abs(actual - expected)}`);
}

// A 3^3 coarse block; selected far-side neighbors subdivide at a 2:1 ratio.
// boundaryAxes makes the home lie on a face, edge or corner of the support.
function support(refinedAxes = 0, boundaryAxes = 0, clipped = false): Donor[] {
  const boxes: Box[] = [];
  for (let iz = -1; iz <= 1; iz++) for (let iy = -1; iy <= 1; iy++) {
    for (let ix = -1; ix <= 1; ix++) {
      const c = [ix, iy, iz].map((x, a) => x + (a < boundaryAxes ? 1 : 0));
      const lo = c.map(x => x - 0.5), hi = c.map(x => x + 0.5);
      if (clipped) for (let a = 0; a < boundaryAxes; a++) lo[a] = Math.max(lo[a], -0.19 - a * 0.07);
      const refine = refinedAxes > 0 && c.slice(0, refinedAxes).every(x => x > 0);
      for (let octant = 0; octant < (refine ? 8 : 1); octant++) {
        boxes.push({ lower: physical(vec(lo.map((x, a) => refine && (octant & (1 << a)) ? c[a] : x))),
          upper: physical(vec(hi.map((x, a) => refine && !(octant & (1 << a)) ? c[a] : x))) });
      }
    }
  }
  // Sparse, deliberately non-positional ordinals, with reversed support order.
  return boxes.map((box, i) => ({ box, id: 1009 + 37 * i })).reverse();
}
function home(donors: Donor[]): Donor {
  return donors.find(d => d.box.lower.every((lo, a) => lo <= frame.origin[a])
    && d.box.upper.every((hi, a) => hi >= frame.origin[a]))!;
}

for (const boundaryAxes of [0, 1, 2, 3]) for (const refinedAxes of [0, 1, 2, 3]) {
  test(`quadratic volume-mean reproduction: boundary axes ${boundaryAxes}, refined axes ${refinedAxes}`, () => {
    const donors = support(refinedAxes, boundaryAxes, boundaryAxes > 0);
    const packet = compileStencil(donors, home(donors).id, frame, 17);
    assert.ok(packet.rankRatio > 1e-3 && Number.isFinite(packet.rankRatio));
    for (let mode = 0; mode < 10; mode++) {
      const expected = Array.from({ length: 10 }, (_, k) => Number(k === mode));
      const means = new Map(donors.map(d => [d.id, boxMean(d.box, p => value(expected, p))]));
      const fit = applyStencil(packet, 17, id => means.get(id)!);
      fit.coefficients.forEach((c, k) => close(c, expected[k]));
      for (const local of [[-0.3, 0.19, 0.4], [1.4, 0.6, 1.7], [0, 0, 0]]) {
        const p = physical(vec(local));
        close(value(fit.coefficients, p), value(expected, p));
      }
    }
  });
}

test("nonpolynomial donor data preserves arbitrary home mean and reads each donor once", () => {
  const donors = support(2, 3, true), h = donors[7];
  const packet = compileStencil(donors, h.id, frame, 3);
  const means = new Map(donors.map((d, i) => [d.id, Math.sin(i * 1.73) + 0.2 * (i % 3)]));
  const reads: number[] = [];
  const fit = applyStencil(packet, 3, id => { reads.push(id); return means.get(id)!; });
  assert.deepEqual(reads, Array.from(packet.ids));
  close(boxMean(h.box, p => value(fit.coefficients, p)), means.get(h.id)!);
  assert.equal(packetCost(packet).densityReadsPerApply, donors.length);
  assert.equal(packetCost(packet).multiplyAddsPerApply, 10 * donors.length);
  assert.equal(packetCost(packet).expandedFloat32WeightsBytes, 40 * donors.length);
  assert.equal(packetCost(packet).donorOrdinalBytes, 4 * donors.length);
});

test("f32 emulation bounds reconstruction error for mixed/clipped support", t => {
  let maxCoefficientError = 0, maxHomeError = 0;
  const coefficients = [0.43, 0.23, -0.18, 0.17, 0.011, -0.017, 0.013, -0.019, 0.021, 0.023];
  for (const boundaryAxes of [0, 1, 2, 3]) for (const refinedAxes of [0, 1, 2, 3]) {
    const donors = support(refinedAxes, boundaryAxes, boundaryAxes > 0), h = home(donors);
    const packet = compileStencil(donors, h.id, frame, 8);
    const means = new Map(donors.map(d => [d.id, boxMean(d.box, p => value(coefficients, p))]));
    let reads = 0;
    const fit = applyStencil(packet, 8, id => { reads++; return means.get(id)!; }, true);
    assert.equal(reads, donors.length);
    fit.coefficients.forEach((c, k) => { maxCoefficientError = Math.max(maxCoefficientError, Math.abs(c - coefficients[k])); });
    maxHomeError = Math.max(maxHomeError, Math.abs(boxMean(h.box, p => value(fit.coefficients, p)) - means.get(h.id)!));
  }
  assert.ok(maxCoefficientError < 3e-6, `coefficient error ${maxCoefficientError}`);
  assert.ok(maxHomeError < 3e-6, `home mean error ${maxHomeError}`);
  t.diagnostic(`16 support geometries: max f32 coefficient error=${maxCoefficientError}, home mean error=${maxHomeError}`);
});

test("rejects stale packets before density reads and missing/nonfinite donor values", () => {
  const donors = support(), packet = compileStencil(donors, home(donors).id, frame, 5);
  let reads = 0;
  assert.throws(() => applyStencil(packet, 6, () => { reads++; return 0; }), /stale/);
  assert.equal(reads, 0);
  for (const invalid of [undefined, NaN, Infinity, -Infinity]) {
    assert.throws(() => applyStencil(packet, 5, () => invalid as number), /non-finite donor/);
  }
});

test("rejects invalid support, ordinals, generation, absent home and deficient geometry", () => {
  const donors = support(), homeId = home(donors).id;
  assert.throws(() => compileStencil(donors.slice(0, 9), homeId, frame, 1), /insufficient/);
  assert.throws(() => compileStencil(donors, 0, frame, 1), /home donor is absent/);
  for (const generation of [-1, 0.5, NaN, Infinity]) {
    assert.throws(() => compileStencil(donors, homeId, frame, generation), /invalid generation/);
  }
  for (const id of [donors[1].id, -1, 1.5, 0xffff_ffff, Infinity]) {
    assert.throws(() => compileStencil([{ ...donors[0], id }, ...donors.slice(1)], homeId, frame, 1), /ordinal/);
  }
  const flat = donors.map(d => ({ ...d, box: { lower: vec([d.box.lower[0], d.box.lower[1], 0]),
    upper: vec([d.box.upper[0], d.box.upper[1], 1]) } }));
  assert.throws(() => compileStencil(flat, homeId, frame, 1), /rank-deficient/);
  const identical = donors.map(d => ({ ...d, box: donors[0].box }));
  assert.throws(() => compileStencil(identical, homeId, frame, 1), /rank-deficient/);
  const invalid = [{ ...donors[0], box: { lower: [0, 0, 0] as Vec3, upper: [0, 1, 1] as Vec3 } }, ...donors.slice(1)];
  assert.throws(() => compileStencil(invalid, homeId, frame, 1), /positive-volume/);
});
